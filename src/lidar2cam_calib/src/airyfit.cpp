// namespace robosense_ros {
//     struct EIGEN_ALIGN16 PointXYZIRT_RS_96_AIRY {
//         PCL_ADD_POINT4D;
//         PCL_ADD_INTENSITY;
//         std::uint16_t ring;  // 行号
//         double timestamp;
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
//     };
// }
// // namespace robosense_ros
// POINT_CLOUD_REGISTER_POINT_STRUCT(robosense_ros::PointXYZIRT_RS_96_AIRY,
//                                   (float, x, x)
//                                           (float, y, y)
//                                           (float, z, z)
//                                           // use std::uint32_t to avoid conflicting with pcl::uint32_t
//                                           (float, intensity, intensity)
//                                           (std::uint16_t, ring, ring)
//                                           (double, timestamp, timestamp)
// )

// void ROBOSENSE_AIRY_96_handler(const sensor_msgs::PointCloud2::ConstPtr &msg){
    
    
//     pcl::PointCloud<robosense_ros::PointXYZIRT_RS_96_AIRY> pl_orig;
//     pcl::fromROSMsg(*msg, pl_orig);
//     int plsize = pl_orig.points.size();

//     if(plsize <= 0){
//         std::cout<<"plsize is zero,line:"<<__LINE__<<std::endl;
//         return ;
//     }

//     pcl::PointCloud<PointType>::Ptr pointcloud_edge(new pcl::PointCloud<PointType>());
//     pcl::PointCloud<PointType>::Ptr pointcloud_surf(new pcl::PointCloud<PointType>());

//     pcl::PointCloud<PointType>::Ptr pointcloud_surf_edge_full(new pcl::PointCloud<PointType>());

//     std::sort(pl_orig.begin(), pl_orig.end(), [](const robosense_ros::PointXYZIRT_RS_96_AIRY & a, const robosense_ros::PointXYZIRT_RS_96_AIRY & b)
//     {
//         return a.timestamp < b.timestamp;
//     });


//     double lidar_begin_time = pl_orig.points[0].timestamp;
//     //double lidar_begin_time = msg->header.stamp.toSec();
//     //std::cout<<"msg time:"<<setprecision(16)<<msg->header.stamp.toSec()<<" time1:"<<setprecision(16)<<pl_orig.points[0].timestamp<<" timd2"<<setprecision(16)<<pl_orig.points[pl_orig.points.size()-1].timestamp<<std::endl;
//     //exit(0);
//     for (int i = 0; i < plsize; i++) {
//         PointType added_pt;
//         added_pt.normal_x = 0;
//         added_pt.normal_y = 0;
//         added_pt.normal_z = 0;
//         added_pt.x = pl_orig.points[i].x;
//         added_pt.y = pl_orig.points[i].y;
//         added_pt.z = pl_orig.points[i].z;
//         added_pt.intensity = pl_orig.points[i].intensity;
//         //std::cout<<"added_pt.intensity:"<<added_pt.intensity<<std::endl;
//         //added_pt.curvature = (pl_orig.points[i].timestamp - msg->header.stamp.toSec() + 0.1)* 1000.0;  //ms

//         added_pt.curvature = (pl_orig.points[i].timestamp - lidar_begin_time)* 1000.0;  //ms
//         //std::cout<<"added_pt.curvature:"<<added_pt.curvature<<std::endl;
//         //std::cout<<"pl_orig.points[i].timestamp:"<<std::setprecision(16)<<pl_orig.points[i].timestamp<<std::endl;
//         //std::cout<<"added_pt.curvature:"<<added_pt.curvature<<std::endl;

//         double dist = added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z;
//         //---> x  xi
//         //|
//         //|
//         //
//         //y
//         if ( dist < blind || isnan(added_pt.x) || isnan(added_pt.y) || isnan(added_pt.z) ||added_pt.x!=added_pt.x ||added_pt.y!=added_pt.y ||added_pt.z != added_pt.z)
//             continue;

//         double azimuth = std::atan2(added_pt.x, added_pt.y) * 180.0 / M_PI;
        
        
//         if (azimuth < 80.0 && azimuth > -80.0) {
//             continue;
//         }
        
//         /**
//         // 剔除正后方30度区域（左15°到右15°）
//         // 条件：角度大于165°或小于-165°（即[-180°, -165°] U [165°, 180°]）
//         if (azimuth > 165.0 || azimuth < -165.0) {
//             continue;
//         }

        
//         **/
//         /**
//          if (azimuth < 15.0 && azimuth > 0) {
//             continue;
//         }
//         */
//        pointcloud_surf->points.push_back(added_pt);

//     }
//     *pointcloud_surf_edge_full = *pointcloud_surf;// + *pointcloud_edge;
//     ros::Time time_msg = ros::Time().fromSec(lidar_begin_time);
    
//     pub_func( *pointcloud_surf_edge_full, pub_full, time_msg);
//     pub_func( *pointcloud_surf_edge_full, pub_surf, time_msg );
//     pub_func( *pointcloud_surf_edge_full, pub_corn, time_msg);
    
// }
