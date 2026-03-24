#include "FeatureExtractor.h"
// agatha 为了输出强度信息
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>

// #include <cmath>
// #include <limits>

#include <vector>       // 基于强度时要用到 agatha 01.05
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_set>

FeatureExtractor::FeatureExtractor() {
    // 初始化 EdgeDrawing 对象
    ed = createEdgeDrawing();
    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL; // 使用 Sobel 算子进行边缘检测
    ed->params.GradientThresholdValue = 36; // 梯度阈值
    ed->params.AnchorThresholdValue = 8; // 锚点阈值
    ed->params.Sigma = 1.0; // 高斯模糊的标准差
}

FeatureExtractor::~FeatureExtractor() {
    // 析构函数
}

bool FeatureExtractor::areDisjoint(const PointPair& a, const PointPair& b) {
    return (a.p1 != b.p1 && a.p1 != b.p2 && a.p2 != b.p1 && a.p2 != b.p2);
}

vector<Point2f> FeatureExtractor::findAndSortFourPoints(const vector<Point2f>& centerBuffer) {
    vector<PointPair> pairs;

    // 计算所有点对的距离和中心点
    for (size_t i = 0; i < centerBuffer.size(); ++i) {
        for (size_t j = i + 1; j < centerBuffer.size(); ++j) {
            Point2f p1 = centerBuffer[i];
            Point2f p2 = centerBuffer[j];
            double dist = norm(p1 - p2); // 计算两点之间的欧几里得距离
            Point2f center((p1.x + p2.x) / 2.0f, (p1.y + p2.y) / 2.0f); // 计算中心点
            pairs.push_back({p1, p2, dist, center});
        }
    }

    double minScore = numeric_limits<double>::max();
    vector<Point2f> bestPoints;

    // 遍历所有点对组合，寻找最优的四个点
    for (size_t i = 0; i < pairs.size(); ++i) {
        const PointPair& pair1 = pairs[i];
        for (size_t j = i + 1; j < pairs.size(); ++j) {
            const PointPair& pair2 = pairs[j];
            if (!areDisjoint(pair1, pair2)) continue; // 确保点对不相交

            double distDiff = abs(pair1.distance - pair2.distance); // 距离差
            double yDiff = abs(pair1.center.y - pair2.center.y);    // 垂直方向的中心点差
            double xDiff = abs(pair1.center.x - pair2.center.x);    // 水平方向的中心点差

            if (yDiff < 50) continue; // 最小垂直间距
            if (xDiff > 20) continue; // 最大水平中心偏差
            if (distDiff / (pair1.distance + pair2.distance) > 0.1) continue; // 距离差的相对误差

            double score = distDiff + xDiff + (1000 / yDiff); // 计算评分，越小越优

            if (score < minScore) {
                minScore = score;
                bestPoints = {pair1.p1, pair1.p2, pair2.p1, pair2.p2}; // 更新最优点集
            }
        }
    }

    if (bestPoints.size() != 4) return {}; // 如果未找到四个点，返回空

    // 使用 k-means 按 y 坐标分组，将点分为上下两组
    Mat pointsY(4, 1, CV_32F);
    for (int i = 0; i < 4; ++i) {
        pointsY.at<float>(i) = bestPoints[i].y;
    }

    Mat labels, centers;
    kmeans(pointsY, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    vector<Point2f> upper, lower;
    for (int i = 0; i < 4; ++i) {
        if (labels.at<int>(i) == 0) {
            upper.push_back(bestPoints[i]); // 上组
        } else {
            lower.push_back(bestPoints[i]); // 下组
        }
    }

    if (centers.at<float>(0) > centers.at<float>(1)) {
        swap(upper, lower); // 确保上组在上，下组在下
    }

    // 按 x 坐标对上下两组分别排序
    sort(upper.begin(), upper.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
    sort(lower.begin(), lower.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });

    // 合并排序后的点集
    bestPoints.clear();
    bestPoints.insert(bestPoints.end(), upper.begin(), upper.end());
    bestPoints.insert(bestPoints.end(), lower.begin(), lower.end());

    return bestPoints;
}

vector<Point2f> FeatureExtractor::detectImageEllipseCenters(const Mat& image) {
    if (image.empty()) {
        cerr << "输入图像为空，请检查图像路径或加载方式。" << endl;
        return {};
    }

    // 检测边缘
    ed->detectEdges(image);

    // 检测线条
    ed->detectLines(lines);

    // 检测椭圆
    ed->detectEllipses(ellipses);

    // 提取椭圆中心点
    vector<Point2f> centerBuffer;
    for (size_t i = 0; i < ellipses.size(); i++) {
        Point2f center(static_cast<int>(ellipses[i][0]), static_cast<int>(ellipses[i][1]));
        centerBuffer.push_back(center);
    }
    std::cout<<"==========================>llipses.size():"<<ellipses.size()<<std::endl;
    // 对中心点进行排序
    return findAndSortFourPoints(centerBuffer);
}

vector<PointType> FeatureExtractor::detectLidarEllipseCenters(const PointCloud<PointType>::Ptr &inputCloud) {
    pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/original.pcd", *inputCloud);  // 以二进制格式保存点云到 PCD 文件
    // 过滤点云
    PointCloud<PointType>::Ptr smallCloud = filterPointCloud(inputCloud);
    pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/roi1.pcd", *smallCloud);  // 以二进制格式保存点云到 PCD 文件
    // 降采样
    PointCloud<PointType>::Ptr downSampledCloud(new PointCloud<PointType>);
    VoxelGrid<PointType> voxelGrid;
    voxelGrid.setInputCloud(smallCloud);
    voxelGrid.setLeafSize(0.005f, 0.005f, 0.005f);
    voxelGrid.filter(*downSampledCloud);

    // 平面分割
    SACSegmentation<PointType> segmentation;                 // 创建 RANSAC 分割对象，用于模型拟合（这里是平面）
    PointIndices::Ptr inliers(new PointIndices);             // 用于存储符合模型的内点索引（平面内点）
    ModelCoefficients::Ptr coefficients(new ModelCoefficients); // 用于存储拟合得到的模型参数（平面系数 a,b,c,d）

    segmentation.setOptimizeCoefficients(true);              // 启用模型参数优化，在 RANSAC 找到内点后对模型做最小二乘精修
    segmentation.setModelType(SACMODEL_PLANE);               // 设置拟合模型类型为平面模型
    segmentation.setMethodType(SAC_RANSAC);                  // 设置随机一致性算法为 RANSAC
    segmentation.setDistanceThreshold(0.02);                 // 设置点到模型的距离阈值，小于该值认为是平面内点（单位：米）
    segmentation.setInputCloud(downSampledCloud);            // 设置输入点云（通常为下采样后的点云以提高效率）
    segmentation.segment(*inliers, *coefficients);           // 执行分割，输出平面内点索引和对应的平面模型参数


    if (inliers->indices.empty()) {
        cout << "Could not estimate a planar model for the given dataset." << endl;
        return {};
    }

    // 提取平面点云
    ExtractIndices<PointType> extract;                        // 创建索引提取器，用于根据索引从点云中提取子集
    extract.setInputCloud(downSampledCloud);                  // 设置输入点云（通常为下采样后的点云）
    extract.setIndices(inliers);                              // 设置需要提取的点索引（RANSAC 得到的平面内点）
    extract.setNegative(false);                               // 设置为 false 表示“提取索引对应的点”（true 则表示剔除这些点）
    lock_guard<mutex> segmentedLock(segmentedGroundMutex);    // 加互斥锁，保护 segmentedGround 的线程安全访问
    segmentedGround.reset(new PointCloud<PointType>);         // 创建新的点云对象，用于存储提取后的平面点云
    extract.filter(*segmentedGround);                         // 执行索引提取，将平面内点写入 segmentedGround
    // 输出点云PCD
    pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/segmented_ground2.pcd", *segmentedGround);  // 以二进制格式保存点云到 PCD 文件

    // 平面点云对齐
    PointCloud<PointType>::Ptr alignedCloud(new PointCloud<PointType>);

    Eigen::Vector3d normal(coefficients->values[0],         // 提系数得到法向量
        coefficients->values[1],
        coefficients->values[2]);
    normal.normalize();                                      // 归一化法向量
    Eigen::Vector3d zAxis(0, 0, 1);                          // 定一个 Z 轴单位向量

    Eigen::Vector3d axis = normal.cross(zAxis);              // 计算旋转轴（法向量与 Z 轴的叉积）
    double angle = acos(normal.dot(zAxis));                  // 计算旋转角度（法向量与 Z 轴的夹角）

    Eigen::AngleAxisd rotation(angle, axis);                 // 定义旋转变换
    Eigen::Matrix3d r = rotation.toRotationMatrix();         // 构造旋转矩阵

    // 应用旋转矩阵，将平面对齐到 Z=0 平面
    float averageZ = 0.0;                                    // 计算平面点云的平均 Z 值
    int cnt = 0;                                             // 计数器
    for (const auto &pt : *segmentedGround) {                /** 把每个点转到了z平面*/
        Eigen::Vector3d point(pt.x, pt.y, pt.z);
        Eigen::Vector3d alignedPoint = r * point;
        pcl::PointXYZI pointi;
        pointi.x = static_cast<float>(alignedPoint.x());
        pointi.y = static_cast<float>(alignedPoint.y());
        pointi.z = 0.0f;
        // pointi.intensity = 0.0f;                              // TODO 这里的强度值需要保留
        pointi.intensity = pt.intensity;
        alignedCloud->push_back(pointi);
        averageZ += alignedPoint.z();
        cnt++;
    }
    averageZ /= cnt;
    // TODO 这里开始要改为依赖强度信息进行边缘提取 12.25
    // 保留强度信息图
    pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/alignedCloud3.pcd", *alignedCloud);  // 以二进制格式保存点云到 PCD 文件

    /** 匿名函数*/
    // auto dump_intensity_sorted = [&](const std::string& out_dir) {
    // // 1) 拷贝指针点云到 vector，方便排序（不改原点云）
    // std::vector<pcl::PointXYZI> pts;
    // pts.reserve(alignedCloud->size());
    // for (const auto& p : *alignedCloud) pts.push_back(p);

    // // 2) 按 intensity 从强到弱排序（降序）
    // std::sort(pts.begin(), pts.end(),
    //           [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
    //               return a.intensity > b.intensity;
    //           });

    // // 3) 写文件：每行一个 intensity（也可顺便把 xyz 写上）
    // const std::string out_path = out_dir + "/intense_list.txt";
    // std::ofstream ofs(out_path, std::ios::out);
    // if (!ofs.is_open()) {
    //     throw std::runtime_error("Failed to open: " + out_path);
    // }

    // for (const auto& p : pts) {
    //     ofs << p.intensity << "\n";
    //     // 如果你想同时输出坐标，改成：
    //     // ofs << p.intensity << " " << p.x << " " << p.y << " " << p.z << "\n";
    // }
    // ofs.close();
    // };

    /** 按照位置输出*/
    // // cloud: 你要用“原坐标位置”的点云（如果要用旋转前，就传 segmentedGround；用旋转后就传 alignedCloud）
    // // resolution: 每个像素代表多少米，比如 0.05 = 5cm/格
    // auto dump_intensity_ascii_map = [&](const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    //                                     const std::string& out_path,
    //                                     float resolution = 0.05f) {
    //     if (!cloud || cloud->empty()) {
    //         throw std::runtime_error("cloud is empty");
    //     }
    //     if (resolution <= 0.0f) {
    //         throw std::runtime_error("resolution must be > 0");
    //     }

    //     // 1) 统计 x/y 范围
    //     float minx = std::numeric_limits<float>::infinity();
    //     float maxx = -std::numeric_limits<float>::infinity();
    //     float miny = std::numeric_limits<float>::infinity();
    //     float maxy = -std::numeric_limits<float>::infinity();

    //     for (const auto& p : *cloud) {
    //         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity)) continue;
    //         minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
    //         miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    //     }

    //     // 2) 计算栅格尺寸（至少 1x1）
    //     const int W = std::max(1, static_cast<int>(std::floor((maxx - minx) / resolution)) + 1);
    //     const int H = std::max(1, static_cast<int>(std::floor((maxy - miny) / resolution)) + 1);

    //     // 3) 建图：每格存“最大强度(整数)”，空格子为 -1
    //     std::vector<int> grid(W * H, -1);

    //     auto to_ix = [&](float x) { return static_cast<int>(std::floor((x - minx) / resolution)); };
    //     auto to_iy = [&](float y) { return static_cast<int>(std::floor((y - miny) / resolution)); };

    //     for (const auto& p : *cloud) {
    //         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity)) continue;
    //         int ix = to_ix(p.x);
    //         int iy = to_iy(p.y);
    //         if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;

    //         int inten = static_cast<int>(std::lround(p.intensity));  // 取整数（四舍五入）
    //         int& cell = grid[iy * W + ix];
    //         cell = std::max(cell, inten); // 同一格多个点时取最大强度
    //     }

    //     // 4) 写 txt：为了像“图”，通常 y 从大到小输出（上->下）
    //     std::ofstream ofs(out_path);
    //     if (!ofs.is_open()) {
    //         throw std::runtime_error("Failed to open: " + out_path);
    //     }

    //     // 可选：写个头，方便你知道分辨率和范围（不想要就删）
    //     ofs << "# resolution(m)=" << resolution << " W=" << W << " H=" << H
    //         << " x:[" << minx << "," << maxx << "] y:[" << miny << "," << maxy << "]\n";

    //     for (int y = H - 1; y >= 0; --y) {
    //         for (int x = 0; x < W; ++x) {
    //             int v = grid[y * W + x];
    //             if (v < 0) v = 0; // 空格子写 0（你也可以写 -1）
    //             ofs << v;
    //             if (x != W - 1) ofs << ' ';
    //         }
    //         ofs << "\n";
    //     }
    // };

    /** 输出强度*/
    // dump_intensity_ascii_map(alignedCloud, "/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/intensity_map.txt", 0.05f);
    // dump_intensity_sorted("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src");

    /** 提取边缘点 法向量方式*/
    PointCloud<PointType>::Ptr edgeCloud(new PointCloud<PointType>);

    pcl::NormalEstimation<PointType, pcl::Normal> normalEstimator;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normalEstimator.setInputCloud(alignedCloud);
    normalEstimator.setRadiusSearch(0.03);  // 设置法线估计的搜索半径
    normalEstimator.compute(*normals);      // 计算法线

    pcl::PointCloud<pcl::Boundary> boundaries;
    pcl::BoundaryEstimation<PointType, pcl::Normal, pcl::Boundary> boundaryEstimator;
    boundaryEstimator.setInputCloud(alignedCloud);
    boundaryEstimator.setInputNormals(normals);
    boundaryEstimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
    boundaryEstimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
    boundaryEstimator.compute(boundaries);

    for (size_t i = 0; i < alignedCloud->size(); ++i) {
        if (boundaries.points[i].boundary_point > 0) {
            edgeCloud->push_back(alignedCloud->points[i]);
        }
    }

    // /** 提取边缘点 强度方式*/
    // using CloudI = pcl::PointCloud<pcl::PointXYZI>;

    // CloudI::Ptr edgeCloud(new CloudI);

    // // ===================== 强度边缘提取（栅格 + 差分） =====================
    // auto extract_intensity_edges = [&](const CloudI::Ptr& cloud,
    //                                 float res = 0.05f,        // 栅格分辨率：米/格（建议 0.02~0.10）
    //                                 int thr = 20,             // 强度差阈值：越大边缘越少
    //                                 bool use_max = true) {    // 每格强度聚合：true=max, false=mean
    //     if (!cloud || cloud->empty()) return;

    //     // 1) 统计 x/y 范围
    //     float minx = std::numeric_limits<float>::infinity();
    //     float maxx = -std::numeric_limits<float>::infinity();
    //     float miny = std::numeric_limits<float>::infinity();
    //     float maxy = -std::numeric_limits<float>::infinity();

    //     for (const auto& p : *cloud) {
    //         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity)) continue;        // 忽略无效点
    //         minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
    //         miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    //     }
    //     if (!std::isfinite(minx) || !std::isfinite(miny)) return;

    //     const int W = std::max(1, (int)std::floor((maxx - minx) / res) + 1);
    //     const int H = std::max(1, (int)std::floor((maxy - miny) / res) + 1);

    //     auto idx = [&](int x, int y) { return y * W + x; };     // 一维索引函数
    //     auto to_ix = [&](float x) { return (int)std::floor((x - minx) / res); };
    //     auto to_iy = [&](float y) { return (int)std::floor((y - miny) / res); };

    //     // 2) 栅格强度图：I[y,x]
    //     std::vector<float> I(W * H, std::numeric_limits<float>::quiet_NaN());
    //     std::vector<int>   C(W * H, 0); // count for mean

    //     for (const auto& p : *cloud) {
    //         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity)) continue;
    //         int x = to_ix(p.x), y = to_iy(p.y);
    //         if (x < 0 || x >= W || y < 0 || y >= H) continue;

    //         float& cell = I[idx(x,y)];      // 格子强度
    //         if (use_max) {
    //             if (!std::isfinite(cell)) cell = p.intensity;
    //             else cell = std::max(cell, (float)p.intensity);
    //         } else {
    //             if (!std::isfinite(cell)) cell = 0.0f;
    //             cell += (float)p.intensity;
    //             C[idx(x,y)] += 1;
    //         }
    //     }

    //     if (!use_max) {
    //         for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
    //             int k = idx(x,y);
    //             if (C[k] > 0) I[k] /= (float)C[k];
    //             else I[k] = std::numeric_limits<float>::quiet_NaN();
    //         }
    //     }

    //     // 3) 差分找边缘格子：记录边缘格子的 index
    //     std::vector<uint8_t> edgeMask(W * H, 0);

    //     auto valid = [&](float v){ return std::isfinite(v); };

    //     for (int y = 0; y < H; ++y) {
    //         for (int x = 0; x < W; ++x) {
    //             float v = I[idx(x,y)];
    //             if (!valid(v)) continue;

    //             // 和右邻比较
    //             if (x + 1 < W) {
    //                 float vr = I[idx(x+1,y)];
    //                 if (valid(vr) && std::abs(v - vr) >= thr) {
    //                     edgeMask[idx(x,y)] = 1;
    //                     edgeMask[idx(x+1,y)] = 1;
    //                 }
    //             }
    //             // 和上邻比较
    //             if (y + 1 < H) {
    //                 float vu = I[idx(x,y+1)];
    //                 if (valid(vu) && std::abs(v - vu) >= thr) {
    //                     edgeMask[idx(x,y)] = 1;
    //                     edgeMask[idx(x,y+1)] = 1;
    //                 }
    //             }
    //         }
    //     }

    //     // 4) 回投：把落在边缘格子的点都放进 edgeCloud（强度保留）
    //     edgeCloud->clear();
    //     edgeCloud->reserve(cloud->size()/10);

    //     for (const auto& p : *cloud) {
    //         if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity)) continue;
    //         int x = to_ix(p.x), y = to_iy(p.y);
    //         if (x < 0 || x >= W || y < 0 || y >= H) continue;
    //         if (edgeMask[idx(x,y)]) {
    //             edgeCloud->push_back(p); // intensity 原样保留
    //         }
    //     }
    // };

    // // 调用（你可调 res/thr）：
    // extract_intensity_edges(alignedCloud, 0.005f, 50, true);


    // pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/edgeCloud4.pcd", *edgeCloud);  // 以二进制格式保存点云到 PCD 文件

    // 对边缘点进行聚类
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(edgeCloud);

    vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(0.02); // 设置聚类距离阈值
    ec.setMinClusterSize(50);     // 最小点数
    ec.setMaxClusterSize(1000);   // 最大点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(edgeCloud);
    ec.extract(clusterIndices);     // 执行聚类

    // 对每个聚类进行圆拟合
    vector<PointType> centers;
    Eigen::Matrix3d rInv = r.inverse();
    std::cout << "Detected clusterIndices.size() " << clusterIndices.size() << " clusters." << std::endl;
    for (size_t i = 0; i < clusterIndices.size(); ++i) {
        pcl::PointCloud<PointType>::Ptr cluster(new PointCloud<PointType>);
        for (const auto &idx : clusterIndices[i].indices) {
            cluster->push_back(edgeCloud->points[idx]);
        }

        // 圆拟合
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);      // 存储圆模型的内点索引
        pcl::SACSegmentation<PointType> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_CIRCLE2D);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01); // 设置距离阈值
        seg.setMaxIterations(1000);     // 设置最大迭代次数
        seg.setInputCloud(cluster);
        seg.segment(*inliers, *coefficients);      

        if (!inliers->indices.empty()) {
            // 计算拟合误差
            double error = 0.0;
            for (const auto &idx : inliers->indices) {
                double dx = cluster->points[idx].x - coefficients->values[0];
                double dy = cluster->points[idx].y - coefficients->values[1];
                // double circleRadius = 0.12;    // 预设的圆洞半径   
                // double circleRadius = 0.025;   // 缩减圆洞半径到0.025m = 2.5cm
                // double circleRadius = 0.094;      // 强度信息的圆洞半径
                double circleRadius = 0.049;      // TODO 强度信息的圆洞半径
                double distance = sqrt(dx * dx + dy * dy) - circleRadius; // 距离误差
                error += abs(distance);
            }
            error /= inliers->indices.size();

            // 如果拟合误差较小，则认为是一个圆洞
            if (error < 0.02) {
                // 将恢复后的圆心坐标添加到点云中
                PointType centerPoint;
                centerPoint.x = coefficients->values[0];
                centerPoint.y = coefficients->values[1];
                centerPoint.z = 0.0;

                // 将圆心坐标逆变换回原始坐标系
                Eigen::Vector3d alignedPoint(centerPoint.x, centerPoint.y, centerPoint.z + averageZ);
                Eigen::Vector3d originalPoint = rInv * alignedPoint;

                PointType centerPointOrigin;
                centerPointOrigin.x = originalPoint.x();
                centerPointOrigin.y = originalPoint.y();
                centerPointOrigin.z = originalPoint.z();
                centers.push_back(centerPointOrigin);
            }
            // TODO 对于没有拟合成功的圆 这里没有处理 可以考虑用质心代替 12.25
        }
    }
    centers.size() == 4 ? std::cout << "Successfully detected 4 circle centers." << std::endl
                           : std::cout << "Warning: Detected " << centers.size() << " circle centers instead of 4." << std::endl;
    sortPatternCenters(centers);
    // // 显示centers
    // CloudI::Ptr show_centers(new pcl::PointCloud<pcl::PointXYZI>);
    // for (const auto &pt : centers) {
    //     pcl::PointXYZI pointi;
    //     pointi.x = pt.x;
    //     pointi.y = pt.y;
    //     pointi.z = pt.z;
    //     pointi.intensity = 255.0f;  // 设置一个高强度值以便显示
    //     show_centers->push_back(pointi);
    // }
    // pcl::io::savePCDFileBinary("/home/kilox/workspace/lidar2cam_calib_ws/src/lidar2cam_calib/src/centers.pcd", *show_centers);  // 保存中心点center
    return centers;
}

// 法向量转到 z(0,0,1)平面后 是与Z轴垂直的 2D平面了
// 将四个点按 左上、右上、左下、右下 顺序排序
void FeatureExtractor::sortPatternCenters(vector<PointType>& centers) {
    if (centers.size() != 4) {
        cerr << "Error: The number of centers must be 4." << endl;
        return;
    }

    // 临时数组，存储每个点的索引和对应的z值
    vector<pair<int, double>> zIndices;
    for (size_t i = 0; i < centers.size(); ++i) {
        zIndices.emplace_back(i, centers[i].z);
    }

    // 按照z值排序（从大到小）
    sort(zIndices.begin(), zIndices.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second > b.second;
    });

    // 上组（z最大的两个点）
    vector<int> topIndices = {zIndices[0].first, zIndices[1].first};
    // 下组（z最小的两个点）
    vector<int> bottomIndices = {zIndices[2].first, zIndices[3].first};

    // 在上组中，根据y值排序（y大的是左，y小的是右）
    sort(topIndices.begin(), topIndices.end(), [&](int a, int b) {
        return centers[a].y > centers[b].y;
    });

    // 在下组中，根据y值排序（y大的是左，y小的是右）
    sort(bottomIndices.begin(), bottomIndices.end(), [&](int a, int b) {
        return centers[a].y > centers[b].y;
    });

    // 填充排序后的点
    vector<PointType> sortedCenters(4);
    sortedCenters[0] = centers[topIndices[0]];       // 0: 左上
    sortedCenters[1] = centers[topIndices[1]];       // 1: 右上
    sortedCenters[2] = centers[bottomIndices[0]];    // 2: 左下
    sortedCenters[3] = centers[bottomIndices[1]];    // 3: 右下

    centers = sortedCenters;
}

bool FeatureExtractor::isRectangle(const vector<Point2f>& points, double lengthThreshold) {
    if (points.size() != 4) {
        cerr << "Error: The number of points must be 4." << endl;
        return false;
    }

    vector<double> distances(6); // 6条边和对角线
    int index = 0;

    // 计算所有点对之间的距离
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            distances[index++] = sqrt(dx * dx + dy * dy);
        }
    }

    // 排序距离
    sort(distances.begin(), distances.end());

    // 提取边长和对角线
    vector<double> sides(distances.begin(), distances.begin() + 4);
    vector<double> diagonals(distances.end() - 2, distances.end());

    // 检查边长是否相近
    double maxSideDiff = (*max_element(sides.begin(), sides.end()) - *min_element(sides.begin(), sides.end())) / 
                         (*min_element(sides.begin(), sides.end()));
    std::cout << "2dMaxSideDiff: " << maxSideDiff << std::endl;                 // TODO 这里有个数值   2dMaxSideDiff: 3.25594               
    if (maxSideDiff > 15 * lengthThreshold) {
        return false;
    }

    // 检查对角线是否相近
    double maxDiagDiff = abs(diagonals[0] - diagonals[1]) / min(diagonals[0], diagonals[1]);
    std::cout << "2dMaxDiagDiff: " << maxDiagDiff << std::endl;
    if (maxDiagDiff > lengthThreshold) {
        return false;
    }

    return true;
}

bool FeatureExtractor::isRectangle3D(const vector<PointType>& points, double lengthThreshold) {
    if (points.size() != 4) {
        cerr << "Error: The number of points must be 4." << endl;
        return false;
    }

    vector<double> distances(6); // 6条边和对角线
    int index = 0;

    // 计算所有点对之间的距离
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            double dz = points[i].z - points[j].z;
            distances[index++] = sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    // 排序距离
    sort(distances.begin(), distances.end());

    // 提取边长和对角线
    vector<double> sides(distances.begin(), distances.begin() + 4);     // 前四个是边长
    vector<double> diagonals(distances.end() - 2, distances.end());     // 后两个是对角线

    // 检查边长是否相近
    double maxSideDiff = (*max_element(sides.begin(), sides.end()) - *min_element(sides.begin(), sides.end())) / 
                         (*min_element(sides.begin(), sides.end()));
    std::cout << "3dMaxSideDiff: " << maxSideDiff << std::endl;              // 3dMaxSideDiff: 0.618603       
    if (maxSideDiff > 10 * lengthThreshold) {
        return false;
    }

    // 检查对角线是否相近
    double maxDiagDiff = abs(diagonals[0] - diagonals[1]) / min(diagonals[0], diagonals[1]);
    std::cout << "3dMaxDiagDiff: " << maxDiagDiff << std::endl;
    if (maxDiagDiff > lengthThreshold) {
        return false;
    }

    return true;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr FeatureExtractor::getSegmentedGround() {
    std::lock_guard<std::mutex> segmentedlock(segmentedGroundMutex);
    PointCloud<PointType>::Ptr outputCloud(new PointCloud<PointType>);
    pcl::copyPointCloud(*segmentedGround, *outputCloud);

    return outputCloud;
}

pcl::PointCloud<PointType>::Ptr FeatureExtractor::filterPointCloud(const pcl::PointCloud<PointType>::Ptr& inputCloud) {
    // 创建一个 PassThrough 过滤器 直通滤波器 
    pcl::PassThrough<PointType> pass;
    pcl::PointCloud<PointType>::Ptr filteredCloud(new pcl::PointCloud<PointType>);

    // 过滤 x 轴
    pass.setInputCloud(inputCloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, 2.0);
    pass.filter(*filteredCloud);

    // 过滤 y 轴
    pass.setInputCloud(filteredCloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-2.0, 2.0);
    pass.filter(*filteredCloud);

    // 过滤 z 轴
    pass.setInputCloud(filteredCloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5, 2.0);
    pass.filter(*filteredCloud);
    // TODO 雷达系的原点在哪里
    // 创建一个空的点云对象来存储最终结果
    pcl::PointCloud<PointType>::Ptr resultCloud(new pcl::PointCloud<PointType>);

    // 栅格大小
    float grid_size = 1.0;

    // 创建一个 map 来存储每个栅格的点
    std::map<std::pair<int, int>, std::vector<PointType>> grid_map;

    // 将点分配到栅格
    for (const auto& point : filteredCloud->points) {
        int grid_x = static_cast<int>(point.x / grid_size);
        int grid_y = static_cast<int>(point.y / grid_size);
        grid_map[{grid_x, grid_y}].push_back(point);
    }

    // 遍历每个栅格
    for (const auto& grid : grid_map) {
        const auto& points_in_grid = grid.second;

        // 如果栅格内点数不足，跳过
        if (points_in_grid.size() < 2) {
            continue;
        }

        // 计算栅格内点的高度差
        float min_z = points_in_grid[0].z;
        float max_z = points_in_grid[0].z;
        for (const auto& point : points_in_grid) {
            if (point.z < min_z) min_z = point.z;
            if (point.z > max_z) max_z = point.z;
        }

        // 如果高度差大于0.5，保留这些点
        if (max_z - min_z > 0.5) {
            for (const auto& point : points_in_grid) {
                resultCloud->push_back(point);
            }
        }
    }

    return resultCloud;
    
}