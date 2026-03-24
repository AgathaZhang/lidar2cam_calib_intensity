cmake_minimum_required(VERSION 3.10)
project(lidar2cam_calib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_DIR "/home/kilox/workspace/3rdparty/opencv-4.5.5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
set(cv_bridge_DIR "/home/kilox/workspace/3rdparty/cv_bridge/share/cv_bridge/cmake")
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  pcl_ros
  pcl_conversions
  sensor_msgs
  image_transport
  message_generation
  std_srvs
  std_msgs
  eigen_conversions
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rosbag
    pcl_ros
    pcl_conversions
    sensor_msgs
    image_transport
    std_srvs
    std_msgs
    eigen_conversions
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
)

add_library(FeatureExtractor SHARED src/FeatureExtractor.cpp)
target_link_libraries(FeatureExtractor
    PRIVATE
        ${OpenCV_LIBS}
        ${Eigen3_LIBRARIES}
        ${PCL_LIBRARIES}
)

add_library(Lidar2CamCalib SHARED src/FactorPrior.cpp src/Lidar2CamCalib.cpp)
target_link_libraries(Lidar2CamCalib
    PRIVATE
        FeatureExtractor
        ${OpenCV_LIBS}
        ${Eigen3_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
)

add_executable(RosNode src/RosNode.cpp)
target_link_libraries(RosNode
  Lidar2CamCalib
  FeatureExtractor
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)

# add_executable(test test.cpp)
# target_link_libraries(test
#   Lidar2CamCalib
#   FeatureExtractor
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBS}
#   ${PCL_LIBRARIES}
#   ${Eigen3_LIBRARIES}
#   ${YAML_CPP_LIBRARIES}
# )

install(TARGETS RosNode
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

