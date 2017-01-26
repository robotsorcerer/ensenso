#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <string>


//Forward Declarations
namespace pathfinder
{
  static bool getROSPackagePath(const std::string pkgName, 
  								boost::filesystem::path & pkgPath);

  static bool copyDirectory(const boost::filesystem::path srcPath,
  							 const boost::filesystem::path dstPath);  

  bool cloudsAndImagesPath(boost::filesystem::path & imagesPath, \
  							boost::filesystem::path & cloudsPath, 
  							const std::string& pkgName = "ensenso");

  std::tuple<boost::filesystem::path, const std::string&, const std::string&,
        	const std::string&, const std::string&, const std::string&, 
        	const std::string&> getCurrentPath();
}

namespace generic
{
	/** @brief Gets the orientation quaternion of a mesh model such that the z-axis is 
	 *   normal to the plane, and the x-y
	 *   axis are as close as possible to the the table frame
	 *
	 *  @param[in] PointA  The point cloud of the head
	 *
	 *  @param[out] quaternion: the quaternion that would rotate the origin frame to fit the head model (z-normal)
	 * */
	tf::Quaternion getQuaternionFrom3Vector(Eigen::Vector3d PointA,
	 										Eigen::Vector3d PointB);

	/*!
    * Given two vectors and two integers aligns the Tf to the first vector and makes the other closest to second vector (0,1,2)=(x,y,z)
    *
    * \param vector1       The vector that we are aligning with
    * \param axis1          Defines which axis (x,y, or z?)
    * \param vector2       The vector that we want to be closest to
    * \param axis2          Defines which axis (x,y, or z?)
    *
    * \returns quaternion   The quaternion that aligns axis1 with vector1 and has axis2 as close to vector2 as possible
    * */
	tf::Quaternion getQuaternionFromVectors (const Eigen::Vector3d vector1, 
		const int axis1, const Eigen::Vector3d vector2, const int axis2);
}