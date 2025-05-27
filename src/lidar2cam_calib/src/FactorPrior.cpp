#include "FactorPrior.h"

FactorPrior::FactorPrior(PointType lidarP, Point2f imgP) : lidarP(lidarP), imgP(imgP) {}

FactorPrior::FactorPrior() {}

FactorPrior::~FactorPrior() {}

template <typename T>
bool FactorPrior::operator()(const T* _q, const T* _t, const T* _camera, T* residuals) const {
    Eigen::Quaternion<T> qIncre(_q[3], _q[0], _q[1], _q[2]);
    Eigen::Matrix<T, 3, 1> tIncre(_t[0], _t[1], _t[2]);

    Eigen::Matrix<T, 3, 1> pL(T(lidarP.x), T(lidarP.y), T(lidarP.z));
    Eigen::Matrix<T, 3, 1> pLInC = qIncre.toRotationMatrix() * pL + tIncre;

    Eigen::Matrix<T, 3, 1> pL2d = pLInC;
    T a = pL2d[0] / pL2d[2];
    T b = pL2d[1] / pL2d[2];

    T ud = _camera[0] * a + _camera[2];
    T vd = _camera[1] * b + _camera[3];

    residuals[0] = ud - T(imgP.x);
    residuals[1] = vd - T(imgP.y);

    return true;
}

ceres::CostFunction* FactorPrior::Create(
    PointType lidarP, Point2f imgP) {
    return (new ceres::AutoDiffCostFunction<FactorPrior, 2, 4, 3, 8>(
        new FactorPrior(lidarP, imgP)));
}