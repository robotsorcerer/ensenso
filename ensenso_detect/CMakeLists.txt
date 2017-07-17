cmake_minimum_required(VERSION 2.8.3)
project(ensenso_detect)

# set cmake modules path for ensenso
## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

#ensenso only works with pcl version 1.8.0. Set this to the path where your pclConfig.cmake lives
set(PCL_DIR      /usr/local/share/pcl-1.8/)

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

find_library(PCL /usr/local/lib NO_DEFAULT_PATH)
find_package(catkin REQUIRED COMPONENTS
  ensenso
  image_transport
  roscpp
  PCL
  rospy
  sensor_msgs
)

find_package(Boost COMPONENTS filesystem system serialization
thread REQUIRED)

find_package(OpenCV REQUIRED)

set(ensenso_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${ensenso_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

message(STATUS "INCLUDE_DIRS: " ${ensenso_INCLUDE_DIRS})

include_directories(${ENSENSO_INCLUDE_DIRS}
					 ${PCL_INCLUDE_DIRS}
						)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## Declare executable source files
set(detect_files src/ensenso_detect.cxx)

catkin_package(
	INCLUDE_DIRS include
	 CATKIN_DEPENDS roscpp rospy sensor_msgs message_filters cv_bridge
	 LIBRARIES pathfinder savgol boostsender cv_tools
)

# add_dependencies(bboxes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

macro(add_exec name)
  add_executable(${name} src/${name}.cxx)
  target_link_libraries(${name} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} -lm -ldl)
endmacro()


# add_library(cv_tools src/window_manager.cpp)

# add_dependencies(bboxes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(cv_tools ${catkin_LIBRARIES} ${OpenCV_LIBS}  -lm -ldl)

# macro(add_lib name)
#   add_library(${name} src/${name}.cxx)
#   target_link_libraries(${name} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} -lm -ldl)
# endmacro()

# add_exec(bboxes)
#  add_executable(bboxes src/bboxes.cxx src/window_manager.cpp)
#  target_link_libraries(bboxes ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} -lm -ldl)

