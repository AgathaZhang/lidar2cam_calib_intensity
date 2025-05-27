#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/edge_drawing.hpp>
#include <vector>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/boundary.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <algorithm>
#include <mutex>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace pcl;

typedef PointXYZI PointType;

class FeatureExtractor {
public:
    /**
     * @brief 构造函数，初始化特征提取器
     */
    FeatureExtractor();

    /**
     * @brief 析构函数，释放资源
     */
    ~FeatureExtractor();

    /**
     * @brief 检测图像中的椭圆中心点
     * @param image 输入图像
     * @return 椭圆中心点的坐标列表
     */
    vector<Point2f> detectImageEllipseCenters(const Mat& image);

    /**
     * @brief 检测激光点云中的椭圆中心点
     * @param inputCloud 输入点云
     * @return 椭圆中心点的点云列表
     */
    vector<PointType> detectLidarEllipseCenters(const PointCloud<PointType>::Ptr& inputCloud);

    /**
     * @brief 获取分割后的地面点云
     * @return 分割后的地面点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getSegmentedGround();

    /**
     * @brief 判断点云是否为3D矩形
     * @param points 输入点云
     * @param lengthThreshold 边长误差阈值
     * @return 是否为3D矩形
     */
    bool isRectangle3D(const vector<PointType>& points, double lengthThreshold = 0.05);

    /**
     * @brief 判断点集是否为2D矩形
     * @param points 输入点集
     * @param lengthThreshold 边长误差阈值
     * @return 是否为2D矩形
     */
    bool isRectangle(const vector<Point2f>& points, double lengthThreshold = 0.08);

private:
    /**
     * @brief 辅助结构体：点对
     */
    struct PointPair {
        Point2f p1; ///< 点1
        Point2f p2; ///< 点2
        double distance; ///< 两点之间的距离
        Point2f center; ///< 两点的中心点
    };

    /**
     * @brief 检查两个点对是否不相交
     * @param a 点对a
     * @param b 点对b
     * @return 是否不相交
     */
    bool areDisjoint(const PointPair& a, const PointPair& b);

    /**
     * @brief 查找并排序四个点
     * @param centerBuffer 输入点集
     * @return 排序后的四个点
     */
    vector<Point2f> findAndSortFourPoints(const vector<Point2f>& centerBuffer);

    /**
     * @brief 对中心点进行排序
     * @param centers 输入中心点
     */
    void sortPatternCenters(vector<PointType>& centers);

    /**
     * @brief 过滤点云，移除不符合条件的点
     * @param inputCloud 输入点云
     * @return 过滤后的点云
     */
    pcl::PointCloud<PointType>::Ptr filterPointCloud(const pcl::PointCloud<PointType>::Ptr& inputCloud);

    Ptr<EdgeDrawing> ed; ///< EdgeDrawing 对象，用于边缘检测
    vector<Vec4f> lines; ///< 检测到的线条
    vector<Vec6d> ellipses; ///< 检测到的椭圆
    pcl::PointCloud<pcl::PointXYZI>::Ptr segmentedGround; ///< 分割后的地面点云
    std::mutex segmentedGroundMutex; ///< 线程安全的互斥锁
};

#endif // FEATURE_EXTRACTOR_H