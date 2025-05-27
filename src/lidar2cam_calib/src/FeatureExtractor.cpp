#include "FeatureExtractor.h"

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

    // 对中心点进行排序
    return findAndSortFourPoints(centerBuffer);
}

vector<PointType> FeatureExtractor::detectLidarEllipseCenters(const PointCloud<PointType>::Ptr &inputCloud) {
    // 过滤点云
    PointCloud<PointType>::Ptr smallCloud = filterPointCloud(inputCloud);

    // 降采样
    PointCloud<PointType>::Ptr downSampledCloud(new PointCloud<PointType>);
    VoxelGrid<PointType> voxelGrid;
    voxelGrid.setInputCloud(smallCloud);
    voxelGrid.setLeafSize(0.005f, 0.005f, 0.005f);
    voxelGrid.filter(*downSampledCloud);

    // 平面分割
    SACSegmentation<PointType> segmentation;
    PointIndices::Ptr inliers(new PointIndices);
    ModelCoefficients::Ptr coefficients(new ModelCoefficients);

    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(SACMODEL_PLANE);
    segmentation.setMethodType(SAC_RANSAC);
    segmentation.setDistanceThreshold(0.02);
    segmentation.setInputCloud(downSampledCloud);
    segmentation.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        cout << "Could not estimate a planar model for the given dataset." << endl;
        return {};
    }

    // 提取平面点云
    ExtractIndices<PointType> extract;
    extract.setInputCloud(downSampledCloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    lock_guard<mutex> segmentedLock(segmentedGroundMutex);
    segmentedGround.reset(new PointCloud<PointType>);
    extract.filter(*segmentedGround);

    // 平面点云对齐
    PointCloud<PointType>::Ptr alignedCloud(new PointCloud<PointType>);

    Eigen::Vector3d normal(coefficients->values[0],
        coefficients->values[1],
        coefficients->values[2]);
    normal.normalize();
    Eigen::Vector3d zAxis(0, 0, 1);

    Eigen::Vector3d axis = normal.cross(zAxis);
    double angle = acos(normal.dot(zAxis));

    Eigen::AngleAxisd rotation(angle, axis);
    Eigen::Matrix3d r = rotation.toRotationMatrix();

    // 应用旋转矩阵，将平面对齐到 Z=0 平面
    float averageZ = 0.0;
    int cnt = 0;
    for (const auto &pt : *segmentedGround) {
        Eigen::Vector3d point(pt.x, pt.y, pt.z);
        Eigen::Vector3d alignedPoint = r * point;
        pcl::PointXYZI pointi;
        pointi.x = static_cast<float>(alignedPoint.x());
        pointi.y = static_cast<float>(alignedPoint.y());
        pointi.z = 0.0f;
        pointi.intensity = 0.0f;
        alignedCloud->push_back(pointi);
        averageZ += alignedPoint.z();
        cnt++;
    }
    averageZ /= cnt;

    // 提取边缘点
    PointCloud<PointType>::Ptr edgeCloud(new PointCloud<PointType>);

    pcl::NormalEstimation<PointType, pcl::Normal> normalEstimator;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normalEstimator.setInputCloud(alignedCloud);
    normalEstimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
    normalEstimator.compute(*normals);

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
    ec.extract(clusterIndices);

    // 对每个聚类进行圆拟合
    vector<PointType> centers;
    Eigen::Matrix3d rInv = r.inverse();

    for (size_t i = 0; i < clusterIndices.size(); ++i) {
        pcl::PointCloud<PointType>::Ptr cluster(new PointCloud<PointType>);
        for (const auto &idx : clusterIndices[i].indices) {
            cluster->push_back(edgeCloud->points[idx]);
        }

        // 圆拟合
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
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
                double circleRadius = 0.12;
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
        }
    }

    sortPatternCenters(centers);
    return centers;
}

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
    std::cout << "2dMaxSideDiff: " << maxSideDiff << std::endl;                     
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
    vector<double> sides(distances.begin(), distances.begin() + 4);
    vector<double> diagonals(distances.end() - 2, distances.end());

    // 检查边长是否相近
    double maxSideDiff = (*max_element(sides.begin(), sides.end()) - *min_element(sides.begin(), sides.end())) / 
                         (*min_element(sides.begin(), sides.end()));
    std::cout << "3dMaxSideDiff: " << maxSideDiff << std::endl;                     
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
    // 创建一个 PassThrough 过滤器
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