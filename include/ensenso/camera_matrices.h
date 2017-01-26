
#include <Eigen/Dense>

namespace cam_info
{
	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getLeftCam();

	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getRightCam();
}