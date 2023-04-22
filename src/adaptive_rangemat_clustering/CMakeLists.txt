cmake_minimum_required(VERSION 2.8.3)
project(adaptive_rangemat_clustering)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
)

find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
)

add_executable(adaptive_rangemat_clustering src/adaptive_rangemat_clustering.cpp)
add_dependencies(adaptive_rangemat_clustering ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(adaptive_rangemat_clustering ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})