cmake_minimum_required(VERSION 2.8.3)
project(ensenso_seg)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  tf
  tf_conversions
)

#ensenso only works with pcl version 1.8.0. Set this to the path where your pclConfig.cmake lives
set(PCL_DIR      /usr/local/share/pcl-1.8/)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

find_library(PCL /usr/local/lib NO_DEFAULT_PATH)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  PCL
  roslib
  cv_bridge
  tf
  tf_conversions
	ensenso
)

find_package(Boost COMPONENTS filesystem system serialization
thread REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${ensenso_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

include_directories(${ENSENSO_INCLUDE_DIRS}
					 ${PCL_INCLUDE_DIRS}
						)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(seg_files src/ensenso_segmentation.cxx)

catkin_package(
	INCLUDE_DIRS include
	 CATKIN_DEPENDS roscpp rospy sensor_msgs
	 tf  tf_conversions
	 LIBRARIES pathfinder savgol boostsender
)

add_executable(ensenso_seg    ${seg_files})

target_link_libraries(ensenso_seg
  ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY}
	  -lm -ldl
)
