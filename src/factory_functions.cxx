
#include <ensenso/camera_matrices.h>

namespace cam_info{

	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getLeftCam()
	{
		Eigen::Matrix3d leftKMat, leftRotation;
		Eigen::Matrix4d leftProjection;
		leftKMat << 967.59025686735504, 0, 0,
						0,		967.59025686735504,		0,
				115.33532536063308,	394.56512026860639,	1;

		leftRotation << 	0.99783369813232914,
							0.0027184390085535194,
							-0.065730669865113958,
							-0.0027704131437143527,
							0.999995917693007,
							-0.0006995772535153315,
							0.065728499774245275,
							0.00088016286974392205,
							0.99783715586800525;
		leftProjection << 	1,0,0,0,
							0,	1,0,0,
							0,	0,0,0.0098904973412229247,
						    -115.33532536063308,-394.56512026860639,
						    967.59025686735504,1.7472966090675965;
		return std::make_tuple(leftKMat, leftRotation,leftProjection);

	}

	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getRightCam()
	{
		Eigen::Matrix3d rightKMat, rightRotation;
		Eigen::Matrix4d rightProjection;
		rightKMat << 967.59025686735504, 0, 0,
						0,		967.59025686735504,		0,
				115.33532536063308,	394.56512026860639,	1;

		rightRotation << 	0.99783369813232914,
							0.0027184390085535194,
							-0.065730669865113958,
							-0.0027704131437143527,
							0.999995917693007,
							-0.0006995772535153315,
							0.065728499774245275,
							0.00088016286974392205,
							0.99783715586800525;
		rightProjection << 	1,0,0,0,
							0,	1,0,0,
							0,	0,0,0.0098904973412229247,
						    -115.33532536063308,-394.56512026860639,
						    967.59025686735504,1.7472966090675965;
		return std::make_tuple(rightKMat, rightRotation,rightProjection);

	}
}
namespace generic{


}


