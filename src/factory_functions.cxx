
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
 	double findMax(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, 
 					pcl::PointCloud<pcl::PointXYZ>::Ptr biggestCluster)
 	{
 		auto max = clouds[0]->points.size();
 		for(auto x: clouds)
 		{
 		  if(max <= x->points.size() ) 
 		    {
 		      max = x->points.size();
 	      	  biggestCluster = x;
 		    }
 		}
 		return max;
 	}

 	//save faces points
 	void savefaces(const pcl::PointCloud<pcl::PointXYZ>::Ptr faces)
 	{
 		//Now we write the points to a text file for visualization processing
 		std::ofstream saver("faces.csv", std::ios_base::out);
 		ROS_INFO("Saving faces to face.csv");
 		for(auto i=0; i < faces->points.size(); ++i)
 		{
 			saver << 100*faces->points[i].x << '\t' << 100*faces->points[i].y 
 				  << '\t' << 100*faces->points[i].z << "\n";
 		}
 		saver.close();
 	}

 	//save centroiuds
 	void savepoints(Eigen::Vector4d &centroid)
 	{
 	    //Now we write the points to a text file for visualization processing
 	    std::ofstream midface("pose.csv", std::ios_base::app | std::ios_base::out);
 	    midface << centroid(0) <<"\t" <<centroid(1)<< "\t" << centroid(2) << "\n";
 	    ROS_INFO("Writing %f, %f, %f to pose.csv", centroid(0), centroid(12), centroid(2));
 	    midface.close();
 	}
/*
	Eigen::Vector4d getBackGroundCentroid() const
	{
		//get the centroids of the resulting clusters
		PointCloudTPtr pillows (new PointCloudT);
		getBackGroundCluster();
		pillows = this->pillows;

		//The last compononent of the vector is set to 1, this allows to transform the centroid vector with 4x4 matrices
		Eigen::Vector4d centroid;
		compute3DCentroid (*pillows, centroid);

		return centroid;
	}

	//Here I used Euclidean clustering to get the clouds of the base pillow which happens to be
	the largest cluster in the scene
	void getBackGroundCluster() const
	{	
		// boost::filesystem::path imagesPath, cloudsPath;
		// pathfinder::cloudsAndImagesPath(imagesPath, cloudsPath);

		pcl::io::loadPCDFile<PointT>("background_cloud.pcd", *cloud_background);
		//remove the table from the background so that what we are left with are the pillows
		PointCloudTPtr filteredCloud(new PointCloudT);
		PointCloudTPtr pillows(new PointCloudT);
		PointCloudTPtr cloud_f(new PointCloudT);
		// Get the plane model, if present.
		pcl::VoxelGrid<pcl::PointXYZ> vg;
		vg.setInputCloud(cloud_background);
		vg.setLeafSize (0.01f, 0.01f, 0.01f);
		vg.filter (*filteredCloud);

		pcl::SACSegmentation<pcl::PointXYZ> segmentation;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		PointCloudTPtr table(new PointCloudT);

		segmentation.setOptimizeCoefficients (true);
		segmentation.setModelType(pcl::SACMODEL_PLANE);
		segmentation.setMethodType(pcl::SAC_RANSAC);
		segmentation.setMaxIterations (100);
		segmentation.setDistanceThreshold(0.1);   //height of table from cam center

		int i = 0, nr_points = static_cast<int>(filteredCloud->points.size());
		for(; filteredCloud->points.size()>0.3*nr_points ;)
		{
			segmentation.setInputCloud(filteredCloud);
			segmentation.segment(*inliers, *coefficients);
			if(inliers->indices.size() == 0)
			{
				OUT("Planar model not found");
				break;
			}
			// Extract the planar inliers from the input cloud
			pcl::ExtractIndices<pcl::PointXYZ> extract;
			extract.setInputCloud (filteredCloud);
			extract.setIndices (inliers);
			extract.setNegative (false);

			// Get the points associated with the planar surface
			extract.filter (*table);

			// Remove the planar inliers, extract the rest
			extract.setNegative (true);
			extract.filter (*cloud_f);
			*filteredCloud = *cloud_f;
		}

		// Creating the KdTree object for the search method of the extraction
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
		tree->setInputCloud (filteredCloud);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance (0.02); // 2cm
		ec.setMinClusterSize (50);
		ec.setMaxClusterSize (2500);
		ec.setSearchMethod (tree);
		ec.setInputCloud (filteredCloud);
		ec.extract (cluster_indices);

		int j = 0;
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > clustersVec;
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
		{
		  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		  for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
		    cloud_cluster->points.push_back (filteredCloud->points[*pit]); //*
		  cloud_cluster->width = cloud_cluster->points.size ();
		  cloud_cluster->height = 1;
		  cloud_cluster->is_dense = true;
		  if(print)
		  	std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;		  //my additions
		  clustersVec.push_back(cloud_cluster);
		  j++;
		}
		//find the cluster with the max size and pass it to this->pillows
		pcl::PointCloud<pcl::PointXYZ>::Ptr biggestCluster (new pcl::PointCloud<pcl::PointXYZ>);
		auto max = generic::findMax(clustersVec, biggestCluster);
		std::cout << "biggest Cluster has " << biggestCluster->points.size () << " data points." << std::endl;

		this->pillows = biggestCluster;
	}

 	void passThrough(const PointCloudT& cloud, PointCloudTPtr passThruCloud) const
 	{
 		//create the filter object and assign it to cloud
 		pcl::PassThrough<PointT> filter;
 		PointCloudTPtr temp_cloud (new PointCloudT (cloud));
 		filter.setInputCloud(temp_cloud);

 		filter.setFilterFieldName("y");
 		filter.setFilterLimits(0, 2);
 		filter.filter(*passThruCloud);
 	}

 	void outlierRemoval(const PointCloudTPtr cloud_in, PointCloudTPtr cloud_out) const
 	{
 		pcl::RadiusOutlierRemoval<PointT> rorfilter (true); // Initializing with true will allow us to extract the removed indices
 		rorfilter.setInputCloud (cloud_in);
 		rorfilter.setRadiusSearch (0.8);
 		rorfilter.setMinNeighborsInRadius (15);
 		// rorfilter.setNegative (true);
 		rorfilter.filter (*cloud_out);
 		// The resulting cloud_out contains all points of cloud_in that have 4 or less neighbors within the 0.1 search radius
 		// indices_rem = rorfilter.getRemovedIndices ();
 		// The indices_rem array indexes all points of cloud_in that have 5 or more neighbors within the 0.1 search radius
 	}*/
}


