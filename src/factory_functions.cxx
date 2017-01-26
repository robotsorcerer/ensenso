
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ensenso/camera_matrices.h>
#include <ensenso/ensenso_headers.h>

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
 	tf::Quaternion getQuaternionFrom3Vector(Eigen::Vector3d PointA, Eigen::Vector3d PointB)
 	{
 		double normAXB = PointA.cross(PointB).norm();   //get A cross B
 		double normBXA = PointB.cross(PointA).norm();   //get B cross A
 		double DotAB   = PointA.dot(PointB);

 		//get the manipulation matrix G
 		Eigen::Matrix3d G;
 		G << DotAB, -1*normAXB, 0,
 			 normAXB, DotAB, 0,
 			 0      ,    0 ,  1;

 		Eigen::Vector3d U = (DotAB * PointA).normalized();
	    Eigen::Vector3d V = (PointB - DotAB * PointA).normalized();
	    Eigen::Vector3d W = PointB.cross(PointA);
	    Eigen::Matrix3d Fproto;
	    Fproto << U, V, W; 

	    Eigen::Matrix3d Rotation_Matrix = (Fproto * G * Fproto.inverse()).inverse();

	    Eigen::Quaterniond q(Rotation_Matrix);
	    tf::Quaternion tf_q;
	    tf::quaternionEigenToTF(q, tf_q);

	    return tf_q;
 	}


 	tf::Quaternion getQuaternionFromVectors (const Eigen::Vector3d vector1, const int axis1, const Eigen::Vector3d vector2, const int axis2) {
 	    std::vector<Eigen::Vector3d> world_axis;
 	    Eigen::Vector3d temp, vector1_norm, proj_vector, proj_vectornorm, axis2_initial_norm_eigen;
 	    tf::Vector3 axis2_tf, axis2_initial, axis2_initial_norm;
 	    tf::Quaternion q, qcheck, u;
 	    double angle, anglecheck;
 	    temp = vector1;
 	    world_axis.resize(3);
 	    world_axis[0] << 1, 0, 0;
 	    world_axis[1] << 0, 1, 0;
 	    world_axis[2] << 0, 0, 1;
 	    vector1_norm = temp / temp.norm();
 	    tf::quaternionEigenToTF(Eigen::Quaterniond().setFromTwoVectors(world_axis[axis1], (vector1_norm)), q);
 	    proj_vector = vector2 - ((vector1_norm.dot(vector2)) * vector1_norm);
 	    proj_vectornorm = proj_vector / proj_vector.norm();
 	    tf::vectorEigenToTF(world_axis[axis2], axis2_tf);
 	    axis2_initial = tf::quatRotate(q, axis2_tf);
 	    axis2_initial_norm = axis2_initial.normalized();
 	    tf::vectorTFToEigen(axis2_initial_norm, axis2_initial_norm_eigen);
 	    angle = acos(axis2_initial_norm_eigen.dot(proj_vectornorm));
 	    if (axis1 == 0)
 	        u.setRPY(angle, 0, 0);
 	    if (axis1 == 1)
 	        u.setRPY(0, angle, 0);
 	    if (axis1 == 2)
 	        u.setRPY(0, 0, angle);
 	    qcheck = q * u;
 	    tf::vectorEigenToTF(world_axis[axis2], axis2_tf);
 	    axis2_initial = tf::quatRotate(qcheck, axis2_tf);
 	    axis2_initial_norm = axis2_initial.normalized();
 	    tf::vectorTFToEigen(axis2_initial_norm, axis2_initial_norm_eigen);
 	    anglecheck = acos(axis2_initial_norm_eigen.dot(proj_vectornorm));
 	    if (anglecheck < 0.0000001)
 	        return qcheck;
 	    else {
 	        angle = -angle;
 	        if (axis1 == 0)
 	            u.setRPY(angle, 0, 0);
 	        if (axis1 == 1)
 	            u.setRPY(0, angle, 0);
 	        if (axis1 == 2)
 	            u.setRPY(0, 0, angle);
 	        return q * u;

 	    }
 	}
 	/** @brief find the maximum point within a point cloud 
  	 * @param[in] pointer to pointcloud 
 	 * @return maximum point in the cloud
 	 *
 	 * @warning This algorithm is O(n)
 	 */
 	double findMax(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
 	{
 		double max = cloud_in->points[0].z;
 		for(auto j=1; j < cloud_in->points.size(); ++j)
 		{
 		  if(max < cloud_in->points[j].z ) 
 		    {
 		      max = cloud_in->points[j].z;
 		    }
 		}
 		return max;
 	}
}


