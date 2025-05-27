#ifndef FACTOR_PRIOR_H
#define FACTOR_PRIOR_H

#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>

#include "FeatureExtractor.h"

using namespace std;
using namespace pcl;
using namespace cv;

class FactorPrior {
public:
    FactorPrior(PointType lidarP, Point2f imgP);
    FactorPrior();
    ~FactorPrior();

    template <typename T>
    bool operator()(const T* _q, const T* _t, const T* _camera, T* residuals) const;

    static ceres::CostFunction* Create(PointType lidarP, Point2f imgP);

private:
    PointType lidarP;
    Point2f imgP;
};
#endif // FACTOR_PRIOR_H