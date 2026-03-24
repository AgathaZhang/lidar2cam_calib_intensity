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

    int sizeP = lidarPoints.size();                             // 特征点数量
    Eigen::Matrix3d R = T_LtoC.block(0, 0, 3, 3);               // 初始旋转矩阵
    Eigen::Quaterniond q(R);                                    // 初始四元数
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

    Eigen::Map<Eigen::Quaterniond> Q_LtoC = Eigen::Map<Eigen::Quaterniond>(ext);      // 将 ext[0..3] 映射为四元数，表示 LiDAR → Camera 的旋转（零拷贝，不创建新内存）
    Eigen::Map<Eigen::Vector3d>    P_LinC = Eigen::Map<Eigen::Vector3d>(ext + 4);    // 将 ext[4..6] 映射为三维向量，表示 LiDAR 在 Camera 坐标系下的平移


    ceres::LocalParameterization *qParameterization = new ceres::EigenQuaternionParameterization();// 为四元数参数提供局部参数化（保证更新后仍为单位四元数）
    ceres::Problem problem;                                                                        // 创建一个 Ceres 优化问题对象，用于装配参数块与残差

    problem.AddParameterBlock(ext, 4, qParameterization);                                          // 添加旋转参数块：ext[0..3] 作为四元数（4维），使用四元数局部参数化更新
    problem.AddParameterBlock(ext + 4, 3);                                                         // 添加平移参数块：ext[4..6] 作为三维平移向量（3维）
    problem.AddParameterBlock(camera, 8);                                                          // 添加相机内参/畸变参数块：camera[0..7]（8维）
    problem.SetParameterBlockConstant(camera);                                                     // 将相机参数设为常量（固定不优化），只优化外参 ext


    for (int val = 0; val < sizeP; val++) {                                                             // 遍历所有配对的特征点（第 val 个 3D-2D 对应）
        ceres::CostFunction* costFunction = FactorPrior::Create(lidarPoints[val], imagePoints[val]);    // 为该对特征点构建一个残差项（代价函数），通常是重投影误差/几何误差
        problem.AddResidualBlock(costFunction, NULL, ext, ext + 4, camera);                              // 将残差块加入优化问题：优化变量为旋转 ext、平移 ext+4；camera 也传入（此处已被设为常量）
    }                                                                                                    // 循环结束后，problem 中包含 sizeP 个残差项用于共同约束外参估计

    ceres::Solver::Options options;                                                // 创建 Ceres 求解器配置项，用于设置优化算法与输出行为
    options.linear_solver_type = ceres::DENSE_SCHUR;                               // 选择线性求解器类型为 Dense Schur（适合重投影类 BA/Schur 结构问题）
    options.minimizer_progress_to_stdout = true;                                   // 将每次迭代的优化过程信息输出到终端
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;               // 选择信赖域策略为 LM（Levenberg-Marquardt），提高非线性最小二乘的稳定性
    ceres::Solver::Summary summary;                                                // 用于保存求解结果摘要（迭代次数、收敛情况、最终代价等）
    ceres::Solve(options, &problem, &summary);                                     // 调用 Ceres 执行求解，对 problem 中的参数块（主要是外参 ext）进行非线性优化

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