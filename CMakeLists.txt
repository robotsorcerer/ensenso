cmake_minimum_required(VERSION 2.8.3)
project(ensenso)

# set cmake modules path for ensenso
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# Check for c++11 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	ELSEIF(COMPILER_SUPPORTS_CXX0X)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
    MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

#ensenso only works with pcl version 1.8.0. Set this to the path where your pclConfig.cmake lives
set(PCL_DIR      /usr/local/share/pcl-1.8/)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  PCL
  roslib
  rostime  
  message_filters
  cv_bridge
  image_transport
  pcl_ros
)

find_package(Ensenso)
find_package(MySQL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp rospy sensor_msgs message_filters cv_bridge pcl_ros
)

###########
## Build ##
###########

set(ensenso_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${ensenso_INCLUDE_DIRS}
  ${MYSQL_INCLUDE_DIRECTORIES}
)

message(STATUS "INCLUDE_DIRS: " ${ensenso_INCLUDE_DIRS})

if(ENSENSO_FOUND)
  message(STATUS "Ensenso SDK found")

  include_directories(${ENSENSO_INCLUDE_DIRS}
  					  ${PCL_INCLUDE_DIRS} 
  					  )
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  ## Declare executable source files
  set(viewer_files src/ensenso_viewer.cxx 
  					src/factory_functions.cxx
            src/pathfinder.cxx)

  ## Declare a C++ executable
  add_executable(ensenso_viewer ${viewer_files})

  ## Specify libraries to link a library or executable target against
  target_link_libraries(ensenso_viewer
    ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
  )

else(ENSENSO_FOUND)
	message(WARNING "Ensenso SDK not found. Not building ensenso interface code.")
endif(ENSENSO_FOUND)