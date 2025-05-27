#include "Lidar2CamCalib.h"

Lidar2CamCalib::~Lidar2CamCalib() {}

Lidar2CamCalib::Lidar2CamCalib() {}

Lidar2CamCalib::Lidar2CamCalib(const std::string& configPath) {
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

    // 设置T_LtoC矩阵
    if (config["T_LtoC"]) {
        const YAML::Node& T_LtoC_node = config["T_LtoC"];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                T_LtoC(i, j) = T_LtoC_node[i][j].as<double>();
            }
        }
    } else {
        std::cerr << "T_LtoC not found in config file" << std::endl;
        std::cerr << "Exiting program." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "T_LtoC:" << std::endl << T_LtoC << std::endl;

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

Eigen::Matrix4d Lidar2CamCalib::calExtrinsic(vector<Point2f> imagePoints, vector<PointType> lidarPoints) {

    int sizeP = lidarPoints.size();
    Eigen::Matrix3d R = T_LtoC.block(0, 0, 3, 3);
    Eigen::Quaterniond q(R);
    double ext[7];
    ext[0] = q.x();
    ext[1] = q.y();
    ext[2] = q.z();
    ext[3] = q.w();
    ext[4] = 0;
    ext[5] = 0;
    ext[6] = 0;

    double k1 = cameraDistcoff(0);
    double k2 = cameraDistcoff(1);
    double k3 = cameraDistcoff(2);
    double k4 = cameraDistcoff(3);

    double fx = cameraIntrinsic(0, 0);
    double fy = cameraIntrinsic(1, 1);
    double cx = cameraIntrinsic(0, 2);
    double cy = cameraIntrinsic(1, 2);

    double camera[8];
    camera[0] = fx;
    camera[1] = fy;
    camera[2] = cx;
    camera[3] = cy;
    camera[4] = k1;
    camera[5] = k2;
    camera[6] = k3;
    camera[7] = k4;

    Eigen::Map<Eigen::Quaterniond> Q_LtoC = Eigen::Map<Eigen::Quaterniond>(ext);
    Eigen::Map<Eigen::Vector3d> P_LinC = Eigen::Map<Eigen::Vector3d>(ext + 4);

    ceres::Manifold* qParameterization = new ceres::EigenQuaternionManifold();
    ceres::Problem problem;

    problem.AddParameterBlock(ext, 4, qParameterization);
    problem.AddParameterBlock(ext + 4, 3);
    problem.AddParameterBlock(camera, 8);
    problem.SetParameterBlockConstant(camera);

    for (int val = 0; val < sizeP; val++) {
        ceres::CostFunction* costFunction = FactorPrior::Create(
            lidarPoints[val], imagePoints[val]);
        problem.AddResidualBlock(costFunction, NULL, ext, ext + 4, camera);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Matrix3d R_LtoC = Q_LtoC.toRotationMatrix();
    T_LtoC.block(0, 0, 3, 3) = R_LtoC;
    T_LtoC.topRightCorner(3, 1) = P_LinC;

    return T_LtoC;
}

void Lidar2CamCalib::calculateReprojectionError(
    vector<PointType> lidarPointBuffer,
    Eigen::Matrix3d rotationMatrix,
    Eigen::Vector3d translationVector,
    vector<Point2f>& imagePoints
) {
    vector<Point3d> transPoints;
    vector<Point2d> reprojPoints;

    double fx = cameraIntrinsic(0, 0);
    double fy = cameraIntrinsic(1, 1);
    double cx = cameraIntrinsic(0, 2);
    double cy = cameraIntrinsic(1, 2);

    // 计算投影点
    for (int i = 0; i < lidarPointBuffer.size(); i++) {
        PointType point = lidarPointBuffer[i];
        Eigen::Vector3d pointEigen;
        pointEigen << point.x, point.y, point.z;

        Eigen::Vector3d transformedPoint = rotationMatrix * pointEigen + translationVector;
        transPoints.push_back(Point3d(transformedPoint[0], transformedPoint[1], transformedPoint[2]));

        double normalizedX = transformedPoint[0] / transformedPoint[2];
        double normalizedY = transformedPoint[1] / transformedPoint[2];
        double u = fx * normalizedX + cx;
        double v = fy * normalizedY + cy;

        reprojPoints.push_back(Point2d(u, v));
    }

    // 计算误差
    size_t lidarSize = lidarPointBuffer.size();
    size_t imagePointsSize = imagePoints.size();
    size_t pointCount = min(lidarSize, imagePointsSize);

    double errorSumX = 0.0;
    double errorSumY = 0.0;
    double rmseX = 0.0;
    double rmseY = 0.0;
    double rmseTotal = 0.0;

    for (int i = 0; i < pointCount; i++) {
        errorSumX += pow(reprojPoints[i].x - imagePoints[i].x, 2);
        errorSumY += pow(reprojPoints[i].y - imagePoints[i].y, 2);
    }

    rmseX = sqrt(errorSumX / static_cast<double>(pointCount));
    rmseY = sqrt(errorSumY / static_cast<double>(pointCount));
    rmseTotal = sqrt(pow(rmseX, 2) + pow(rmseY, 2));

    cout << "RMSE Total: " << rmseTotal << endl;
}