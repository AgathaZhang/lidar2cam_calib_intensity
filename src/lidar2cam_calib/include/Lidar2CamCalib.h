#ifndef LIDAR_TO_CAM_CALIB_H
#define LIDAR_TO_CAM_CALIB_H

#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>

#include "FeatureExtractor.h"
#include "FactorPrior.h"

using namespace std;
using namespace pcl;
using namespace cv;

class Lidar2CamCalib {
public:
    Lidar2CamCalib();
    Lidar2CamCalib(const std::string& configPath);
    ~Lidar2CamCalib();
    Eigen::Matrix4d calExtrinsic(vector<Point2f> imagePoints, vector<PointType> lidarPoints);
    void calculateReprojectionError(
        vector<PointType> lidarPointBuffer,
        Eigen::Matrix3d rotationMatrix,
        Eigen::Vector3d translationVector,
        vector<Point2f>& imagePoints
    );

private:
    Eigen::Matrix4d T_LtoC;
    Eigen::Matrix3d cameraIntrinsic;
    Eigen::Vector4d cameraDistcoff;

};
#endif // LIDAR_TO_CAM_CALIB_H