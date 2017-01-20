#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <string>


//Forward Declarations
namespace pathfinder
{
  static bool getROSPackagePath(const std::string pkgName, boost::filesystem::path & pkgPath);
  static bool copyDirectory(const boost::filesystem::path srcPath, const boost::filesystem::path dstPath);  
  bool cloudsAndImagesPath(boost::filesystem::path & imagesPath, boost::filesystem::path & cloudsPath, const std::string& pkgName = "ensenso");
  std::tuple<boost::filesystem::path, const std::string&, const std::string&,\
        const std::string&, const std::string&, const std::string&, \
        const std::string&> getCurrentPath();
}
