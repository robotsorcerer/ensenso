#ifndef _ENSENSO_HEADERS_H_
#define _ENSENSO_HEADERS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <string>

int getOpenCVType (std::string type)
{
  if (type == "CV_32FC1")
    return CV_32FC1;
  else if (type == "CV_32FC2")
    return CV_32FC2;
  else if (type == "CV_32FC3")
    return CV_32FC3;
  else if (type == "CV_32FC4")
    return CV_32FC4;
  else if (type == "CV_64FC1")
    return CV_64FC1;
  else if (type == "CV_64FC2")
    return CV_64FC2;
  else if (type == "CV_64FC3")
    return CV_64FC3;
  else if (type == "CV_64FC4")
    return CV_64FC4;
  else if (type == "CV_8UC1")
    return CV_8UC1;
  else if (type == "CV_8UC2")
    return CV_8UC2;
  else if (type == "CV_8UC3")
    return CV_8UC3;
  else if (type == "CV_8UC4")
    return CV_8UC4;
  else if (type == "CV_16UC1")
    return CV_16UC1;
  else if (type == "CV_16UC2")
    return CV_16UC2;
  else if (type == "CV_16UC3")
    return CV_16UC3;
  else if (type == "CV_16UC4")
    return CV_16UC4;
  else if (type == "CV_32SC1")
    return CV_32SC1;
  else if (type == "CV_32SC2")
    return CV_32SC2;
  else if (type == "CV_32SC3")
    return CV_32SC3;
  else if (type == "CV_32SC4")
    return CV_32SC4;

  return (-1);
}

//Forward Declarations
namespace pathfinder
{
  static bool getROSPackagePath(const std::string pkgName, boost::filesystem::path & pkgPath);
  static bool copyDirectory(const boost::filesystem::path srcPath, const boost::filesystem::path dstPath);
  std::tuple<boost::filesystem::path, const std::string&, const std::string&,\
        const std::string&, const std::string&, const std::string&, \
        const std::string&> getCurrentPath();
}

#endif
