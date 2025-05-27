#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Lidar2CamCalib.h"
#include "FeatureExtractor.h"

using namespace std;
using message_filters::sync_policies::ApproximateTime;
typedef ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> SyncPolicy;

/**
 * @brief ROS节点类，用于处理激光雷达和摄像头数据
 */
class RosNode {
public:
    /**
     * @brief 构造函数，初始化ROS节点
     * @param configPath 配置文件路径
     */
    RosNode(const string& configPath) {
        
        setParameters(configPath);

        calib = std::make_shared<Lidar2CamCalib>(configPath);
        nh = ros::NodeHandle("~");

        // 订阅激光雷达数据
        laserSub = nh.subscribe(lidarTopic, 10, &RosNode::laserCallback, this);

        // 发布累积点云数据
        accumulatedLaserPub = nh.advertise<sensor_msgs::PointCloud2>("/livox/lidar/sum", 10);

        segmentedCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/segmentedcloud", 10);

        featureCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/featurecloud", 10);

        marker_pub = nh.advertise<visualization_msgs::Marker>("/featurecloud_id", 10);
        

        // 初始化图像传输
        image_transport::ImageTransport it(nh);
        imagePub = it.advertise("/usb_cam_left/image_undis", 1);
        feature_imagePub = it.advertise("/feature_image", 1);

        // 订阅摄像头原始图像
        imageSub = it.subscribe(cameraTopic, 1, &RosNode::imageCallback, this);

        accumulatedLaserSub.subscribe(nh, "/livox/lidar/sum", 10);
        undisImageSub.subscribe(nh, "/usb_cam_left/image_undis", 10);
        sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), accumulatedLaserSub, undisImageSub));
        sync->registerCallback(boost::bind(&RosNode::syncCallback, this, _1, _2));

        ROS_INFO("Subscribing to lidar topic: %s", lidarTopic.c_str());
        ROS_INFO("Subscribing to camera topic: %s", cameraTopic.c_str());
    }

    void setParameters(const string& configPath) {
        // 读取YAML文件
        YAML::Node config;
        try {
            config = YAML::LoadFile(configPath);
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading config file: " << e.what() << std::endl;
            std::cerr << "Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }

        // 检查文件是否正确加载
        if (!config) {
            std::cerr << "Failed to load config file: " << configPath << std::endl;
            std::cerr << "Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }

        // 读取话题名称
        if (config["topics"]) {
            const YAML::Node& topics_node = config["topics"];
            lidarTopic = topics_node["lidar"].as<string>();
            cameraTopic = topics_node["camera"].as<string>();
        } else {
            std::cerr << "Topics not found in config file" << std::endl;
            std::cerr << "Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }

        // 设置cameraIntrinsic矩阵
        if (config["cameraIntrinsic"]) {
            const YAML::Node& cameraIntrinsic_node = config["cameraIntrinsic"];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cameraIntrinsic(i, j) = cameraIntrinsic_node[i][j].as<double>();
                }
            }
        } else {
            std::cerr << "cameraIntrinsic not found in config file" << std::endl;
            std::cerr << "Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "cameraIntrinsic:" << std::endl << cameraIntrinsic << std::endl;

        // 设置cameraDistcoff向量
        if (config["cameraDistcoff"]) {
            const YAML::Node& cameraDistcoff_node = config["cameraDistcoff"];
            for (int i = 0; i < 4; ++i) {
                cameraDistcoff(i) = cameraDistcoff_node[i].as<double>();
            }
        } else {
            std::cerr << "cameraDistcoff not found in config file" << std::endl;
            std::cerr << "Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }
    
        std::cout << "cameraDistcoff:" << std::endl << cameraDistcoff << std::endl;
    }

    /**
     * @brief 同步回调函数，处理点云和图像数据
     * @param laserMsg 点云消息
     * @param imageMsg 图像消息
     */
    void syncCallback(const sensor_msgs::PointCloud2::ConstPtr& laserMsg, const sensor_msgs::ImageConstPtr& imageMsg) {
        if (!calibStart) {
            // 点云处理
            pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*laserMsg, *inputCloud);
    
            vector<PointXYZI> lidarCenters = extractor.detectLidarEllipseCenters(inputCloud);
            pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedGround = extractor.getSegmentedGround();
            pcl::PointCloud<pcl::PointXYZI>::Ptr featureGround(new pcl::PointCloud<pcl::PointXYZI>);
            // 遍历 vector 并将点添加到 PCL 点云中
            for (const auto& point : lidarCenters) {
                featureGround->push_back(point);
            }
            segmentedGround->header = inputCloud->header;
            featureGround->header = inputCloud->header;

            // 发布分割后的点云
            sensor_msgs::PointCloud2 segmentedCloudMsg, featureCloudMsg;
            pcl::toROSMsg(*segmentedGround, segmentedCloudMsg);
            pcl::toROSMsg(*featureGround, featureCloudMsg);
            segmentedCloudPub.publish(segmentedCloudMsg);
            featureCloudPub.publish(featureCloudMsg);

            // 添加序号标记（marker）用于 RViz 可视化
            visualization_msgs::Marker marker;
            marker.header.frame_id = inputCloud->header.frame_id;
            marker.header.stamp = ros::Time::now();
            marker.ns = "ellipse_center_ids";
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING; // 文字始终面向观察者
            marker.id = 0;

            // 设置 marker 的文字颜色和大小
            marker.color.r = 128.0 / 255.0;
            marker.color.g = 0.0;
            marker.color.b = 128.0 / 255.0;
            marker.color.a = 1.0;
            marker.scale.z = 0.2; // 文字大小，根据需要调整

            // 遍历点云中心点并创建 marker
            for (size_t i = 0; i < lidarCenters.size(); ++i) {
                marker.pose.position.x = lidarCenters[i].x;
                marker.pose.position.y = lidarCenters[i].y;
                marker.pose.position.z = lidarCenters[i].z + 0.1; // 将文字稍微抬高，避免与点重叠
                marker.text = std::to_string(i); // 序号
                marker_pub.publish(marker); // 发布 marker
                marker.id++; // 更新 marker ID，避免覆盖
            }

    
            // 检查点云特征是否形成矩形
            if (extractor.isRectangle3D(lidarCenters)) {
                cout << "Lidar centers form a rectangle." << endl;
                cout << "Lidar ellipse centers size: " << lidarCenters.size() << endl;
                for (size_t i = 0; i < lidarCenters.size(); ++i) {
                    cout << "Lidar center " << i << ": (" 
                         << lidarCenters[i].x << ", " 
                         << lidarCenters[i].y << ", " 
                         << lidarCenters[i].z << ")" << endl;
                }
                // 缓冲区管理
                if (lidarCentersBuff.size() / 4 < collectionSize) {
                    lidarCentersBuff.insert(lidarCentersBuff.end(), lidarCenters.begin(), lidarCenters.end());
                }
            } else {
                cout << "Lidar centers do not form a rectangle." << endl;
            }
    
            // 图像处理
            cv::Mat image = cv_bridge::toCvCopy(imageMsg, sensor_msgs::image_encodings::BGR8)->image;
            cv::Mat gray_image;
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
            vector<cv::Point2f> imageCenters = extractor.detectImageEllipseCenters(gray_image);

            cv::Mat feature_image = image.clone();
            for (size_t i = 0; i < imageCenters.size(); ++i) {
                const auto& center = imageCenters[i];
                cv::circle(feature_image, center, 5, cv::Scalar(0, 255, 0), -1);
                // 绘制点的序号
                cv::putText(feature_image, std::to_string(i), 
                            cv::Point(center.x + 10, center.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 0, 255), 2);
            }

            cv_bridge::CvImage cvImage;
            cvImage.header = imageMsg->header;
            cvImage.header.frame_id = "camera";
            cvImage.encoding = sensor_msgs::image_encodings::BGR8;
            cvImage.image = feature_image;

            feature_imagePub.publish(cvImage.toImageMsg());
    
            // 检查图像特征是否形成矩形
            if (extractor.isRectangle(imageCenters)) {
                cout << "Image centers form a rectangle." << endl;
                cout << "Image ellipse centers size: " << imageCenters.size() << endl;
                for (size_t i = 0; i < imageCenters.size(); ++i) {
                    cout << "Image center " << i << ": (" 
                         << imageCenters[i].x << ", " 
                         << imageCenters[i].y << ")" << endl;
                }
                // 缓冲区管理
                if (imageCentersBuff.size() / 4 < collectionSize) {
                    imageCentersBuff.insert(imageCentersBuff.end(), imageCenters.begin(), imageCenters.end());
                }
            } else {
                cout << "Image centers do not form a rectangle." << endl;
            }
    
            // 打印缓冲区状态
            cout << "Image Features Collected: " << imageCentersBuff.size() / 4 << " / " << collectionSize << endl;
            cout << "Lidar Features Collected: " << lidarCentersBuff.size() / 4 << " / " << collectionSize << endl;
        } else {
            cout << "Wait Calib!" << endl;
        }
    
        // 标定逻辑
        if (imageCentersBuff.size() / 4 == collectionSize && lidarCentersBuff.size() / 4 == collectionSize && !calibStart) {
            cout << "Calib Start!" << endl;
            calibStart = true;
    
            vector<cv::Point2f> imageAvgCenters(4);
            vector<PointXYZI> lidarAvgCenters(4);
    
            // 计算平均值
            for (size_t i = 0; i < imageCentersBuff.size(); i += 4) {
                for (int j = 0; j < 4; ++j) {
                    lidarAvgCenters[j].x += lidarCentersBuff[i + j].x;
                    lidarAvgCenters[j].y += lidarCentersBuff[i + j].y;
                    lidarAvgCenters[j].z += lidarCentersBuff[i + j].z;
                    imageAvgCenters[j].x += imageCentersBuff[i + j].x;
                    imageAvgCenters[j].y += imageCentersBuff[i + j].y;
                }
            }

            for (int j = 0; j < 4; ++j) {
                lidarAvgCenters[j].x /= (lidarCentersBuff.size() / 4);
                lidarAvgCenters[j].y /= (lidarCentersBuff.size() / 4);
                lidarAvgCenters[j].z /= (lidarCentersBuff.size() / 4);
                imageAvgCenters[j].x /= (imageCentersBuff.size() / 4);
                imageAvgCenters[j].y /= (imageCentersBuff.size() / 4);
            }

            for (size_t i = 0; i < imageAvgCenters.size(); ++i) {
                cout << "Image center " << i << ": (" 
                     << imageAvgCenters[i].x << ", " 
                     << imageAvgCenters[i].y << ")" << endl;
            }

            for (size_t i = 0; i < lidarAvgCenters.size(); ++i) {
                cout << "Lidar center " << i << ": (" 
                     << lidarAvgCenters[i].x << ", " 
                     << lidarAvgCenters[i].y << ", " 
                     << lidarAvgCenters[i].z << ")" << endl;
            }

            lidarAvgCentersBuff.insert(lidarAvgCentersBuff.end(), lidarAvgCenters.begin(), lidarAvgCenters.end());
            imageAvgCentersBuff.insert(imageAvgCentersBuff.end(), imageAvgCenters.begin(), imageAvgCenters.end());
            // 询问用户是否开始标定
            char userChoice;
            cout << "Image Feature Size: " << imageAvgCentersBuff.size() / 4 << endl;
            cout << "Lidar Feature Size: " << lidarAvgCentersBuff.size() / 4 << endl;
            cout << "Do you want to start calibration now? (y/n): ";
            cin >> userChoice;

            if (userChoice == 'y' || userChoice == 'Y') {
                // 使用当前的平均值进行标定
                Eigen::Matrix4d T_LtoC = calib->calExtrinsic(imageAvgCentersBuff, lidarAvgCentersBuff);
                cout << "Extrinsic Calibration Matrix:" << endl << T_LtoC << endl;
                calib->calculateReprojectionError(lidarAvgCentersBuff, T_LtoC.block(0, 0, 3, 3), T_LtoC.block(0, 3, 3, 1), imageAvgCentersBuff);
                exit(0);
            } else if (userChoice == 'n' || userChoice == 'N') {
                // 继续收集特征
                cout << "Continuing feature collection..." << endl;
                imageCentersBuff.clear();  // 清空图像特征缓冲区
                lidarCentersBuff.clear();  // 清空点云特征缓冲区
                calibStart = false;
            } else {
                cout << "Invalid input. Continuing feature collection..." << endl;
                calibStart = false;
            }
        }
    }

    /**
     * @brief 激光雷达回调函数，处理点云数据
     * @param msg 点云消息
     */
    void laserCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        laserBuffer.push_back(cloud);

        if (laserBuffer.size() >= maxBufferSize) {
            processAndPublish();
            laserBuffer.clear();
        }
    }

    /**
     * @brief 图像回调函数，处理摄像头图像
     * @param msg 图像消息
     */
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            cv::Mat undistorted = undistortImage(cvPtr->image);

            cv_bridge::CvImage cvImage;
            cvImage.header = msg->header;
            cvImage.header.frame_id = "camera";
            cvImage.encoding = sensor_msgs::image_encodings::BGR8;
            cvImage.image = undistorted;

            imagePub.publish(cvImage.toImageMsg());
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    /**
     * @brief 处理并发布累积点云数据
     */
    void processAndPublish() {
        if (laserBuffer.empty()) {
            ROS_WARN("No data in buffer. Skipping processing.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr accumulatedCloud(new pcl::PointCloud<pcl::PointXYZI>);

        for (auto& cloud : laserBuffer) {
            *accumulatedCloud += *cloud;
        }

        accumulatedCloud->header = laserBuffer.back()->header;

        sensor_msgs::PointCloud2 accumulatedMsg;
        pcl::toROSMsg(*accumulatedCloud, accumulatedMsg);
        accumulatedLaserPub.publish(accumulatedMsg);
    }

    /**
     * @brief 对图像进行畸变校正
     * @param distortedImage 畸变图像
     * @return 校正后的图像
     */
    cv::Mat undistortImage(const cv::Mat& distortedImage) {
        double fx = cameraIntrinsic(0, 0);
        double fy = cameraIntrinsic(1, 1);
        double cx = cameraIntrinsic(0, 2);
        double cy = cameraIntrinsic(1, 2);

        double k1 = cameraDistcoff(0);
        double k2 = cameraDistcoff(1);
        double k3 = cameraDistcoff(2);
        double k4 = cameraDistcoff(3);

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat D = (cv::Mat_<double>(4, 1) << k1, k2, k3, k4);

        cv::Mat mapX, mapY;
        cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), K, 
                                          distortedImage.size(), CV_32FC1, mapX, mapY);

        cv::Mat undistortedImage;
        cv::remap(distortedImage, undistortedImage, mapX, mapY, cv::INTER_LINEAR);

        return undistortedImage;
    }

private:
    ros::NodeHandle nh;  //!< ROS节点句柄
    ros::Subscriber laserSub;  //!< 激光雷达订阅器
    image_transport::Publisher imagePub;  //!< 图像发布器
    image_transport::Publisher feature_imagePub;  //!< 图像特征发布器
    image_transport::Subscriber imageSub;  //!< 图像订阅器
    ros::Publisher accumulatedLaserPub;  //!< 累积点云发布器
    ros::Publisher segmentedCloudPub;
    ros::Publisher featureCloudPub;
    ros::Publisher marker_pub;  //!< 标记发布器
    vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laserBuffer;  //!< 点云缓冲区
    size_t maxBufferSize = 30;  //!< 最大缓冲区大小
    std::shared_ptr<Lidar2CamCalib> calib;  //!< 激光雷达到摄像头标定
    Eigen::Matrix3d cameraIntrinsic;
    Eigen::Vector4d cameraDistcoff;
    string lidarTopic;
    string cameraTopic;
    message_filters::Subscriber<sensor_msgs::PointCloud2> accumulatedLaserSub;
    message_filters::Subscriber<sensor_msgs::Image> undisImageSub;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;
    FeatureExtractor extractor;
    vector<cv::Point2f> imageCentersBuff;
    vector<PointXYZI> lidarCentersBuff;
    vector<cv::Point2f> imageAvgCentersBuff;
    vector<PointXYZI> lidarAvgCentersBuff;
    bool calibStart = false;
    int collectionSize = 1;
};

/**
 * @brief 主函数，初始化ROS节点并进入循环
 * @param argc 参数个数
 * @param argv 参数列表
 * @return 退出码
 */
int main(int argc, char** argv) {
    ros::init(argc, argv, "ros_node");

    std::string config_path;
    if (argc < 2) {
        ROS_ERROR("No config file path provided. Usage: %s <config_path>", argv[0]);
        return -1;
    }
    config_path = argv[1];

    RosNode node(config_path);
    ros::spin();
    return 0;
}