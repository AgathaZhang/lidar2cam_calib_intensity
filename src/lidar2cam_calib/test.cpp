#include "Lidar2CamCalib.h"
#include "FeatureExtractor.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace cv;
using namespace std;
using namespace pcl;

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <pcdFilePath> <imagePath>" << endl;
        return -1;
    }

    // 加载图像和点云文件路径
    string pcdFilePath = argv[1];
    string imagePath = argv[2];

    // 加载图像
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: 无法加载图片，请检查路径是否正确。" << endl;
        return -1;
    }

    // 加载点云
    PointCloud<PointXYZI>::Ptr inputCloud(new PointCloud<PointXYZI>);
    if (io::loadPCDFile<PointXYZI>(pcdFilePath, *inputCloud) == -1) {
        cerr << "Error: Failed to load point cloud file!" << endl;
        return -1;
    }

    // 创建特征提取器
    FeatureExtractor extractor;

    // 检测图像中的椭圆中心
    vector<Point2f> imageCenters = extractor.detectImageEllipseCenters(image);

    // 打印图像椭圆中心
    cout << "Image ellipse centers size: " << imageCenters.size() << endl;
    for (size_t i = 0; i < imageCenters.size(); ++i) {
        cout << "Image center " << i << ": (" 
             << imageCenters[i].x << ", " 
             << imageCenters[i].y << ")" << endl;
    }

    // 检测点云中的椭圆中心
    vector<PointXYZI> lidarCenters = extractor.detectLidarEllipseCenters(inputCloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedGround = extractor.getSegmentedGround();

    // 打印点云椭圆中心
    cout << "Lidar ellipse centers size: " << lidarCenters.size() << endl;
    for (size_t i = 0; i < lidarCenters.size(); ++i) {
        cout << "Lidar center " << i << ": (" 
             << lidarCenters[i].x << ", " 
             << lidarCenters[i].y << ", " 
             << lidarCenters[i].z << ")" << endl;
    }

    if (extractor.isRectangle(imageCenters)) {
        cout << "Image centers form a rectangle." << endl;
    } else {
        cout << "Image centers do not form a rectangle." << endl;
    }

    if (extractor.isRectangle3D(lidarCenters)) {
        cout << "Lidar centers form a rectangle." << endl;
    } else {
        cout << "Lidar centers do not form a rectangle." << endl;
    }

    Lidar2CamCalib Calib("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/config/config.yaml");
    Eigen::Matrix4d T_LtoC;
    T_LtoC = Calib.calExtrinsic(imageCenters, lidarCenters);
    std::cout << T_LtoC <<std::endl;
    
    return 0;
}