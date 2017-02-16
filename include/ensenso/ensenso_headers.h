
#include <string>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/*
Without defining boost scoped enums, linking breaks the compilation. 
See http://stackoverflow.com/questions/15634114/cant-link-program-using-boost-filesystem
*/
#define BOOST_NO_CXX11_SCOPED_ENUMS		
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/tuple/tuple.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "tf_conversions/tf_eigen.h"
#include <tf/tf.h>


using PointT      = pcl::PointXYZ;
using PointCloudT     = pcl::PointCloud<PointT>;
using PointCloudTPtr  = PointCloudT::Ptr;

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

  bool getDataDirectory(boost::filesystem::path data_dir);
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

  /* @brief given an angle in radians, it converts it to degrees
  *  @param[in] phi angle in radians
  *  @param[out] phi angle in degrees
  *  @warning, the nagle is changed in place 
  * @note thatthe function is defined here because gnu coompilers 
  * do not do whole optizations
  */
  inline void rad2deg(double& phi)
  {
    phi *= (180/M_PI);
  };

  void savefaces(const pcl::PointCloud<pcl::PointXYZ>::Ptr faces);
  void savepoints(Eigen::Vector4d &centroid);
  /** @brief find the maximum point within a point cloud 
   * @param[in] pointer to pointcloud 
   * @return maximum point in the cloud
   *
   * @warning This algorithm is O(n)
   */
  double findMax(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, 
          pcl::PointCloud<pcl::PointXYZ>::Ptr biggestCluster);

  Eigen::Vector4d findMaxPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds);
}

class sender;