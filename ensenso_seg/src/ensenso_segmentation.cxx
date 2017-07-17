#include <mutex>
#include <cmath>
#include <thread>
#include <memory>
#include <queue>
#include <fstream>
#include <ros/spinner.h>
#include <sensor_msgs/PointCloud2.h>
#include <unsupported/Eigen/AlignedVector3>

#include <ensenso/visualizer.h>
#include <ensenso/pcl_headers.h>
#include <ensenso/savgol.h>
#include <ensenso/camera_matrices.h>
#include <ensenso/ensenso_headers.h>
#include <ensenso/boost_sender.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_svd.h>


/*Computer Descriptors*/
#include <pcl/console/time.h>   // TicToc
#include <pcl/console/parse.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

class objectPoseEstim; //class forward declaration
class sender; //class forward declaration of boost broadcaster
// pcl::registration::TransformationEstimation<PointT, PointT, double> trans_est;

class Segmentation
{
private:
	friend class objectPoseEstim; //friend class forward declaration
	bool updateCloud, save, running, print, savepcd, send, \
		firstFace, next_iteration;
	ros::NodeHandle nh_;
	std::ostringstream oss;
 	const std::string cloudName;
	ros::Subscriber cloud_sub_;
	std::mutex mutex;
 	std::thread cloudDispThread, normalsDispThread;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	std::vector<std::thread> threads;
 	unsigned long const hardware_concurrency;
	float distThreshold; //distance at which to apply the threshold
	float zmin, zmax;	//minimum and maximum distance to extrude the prism object
	int v1, v2, v3, v4;
	int iterations; //# of icp iterations
 	ros::AsyncSpinner spinner;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer;

	// base cloud of scene
	PointCloudT cloud;
	mutable PointCloudTPtr pillows;   //cause we'll copy indices of pillows to this
	/*Plane Segmentation Objects*/
	PointCloudTPtr segCloud, plane, convexHull, faces,
					passThroughFaces, firstFaceCloud, facesOnly,
					facesOnly2, passThruCloud, largest_cluster,
					outlierRemCloud, filteredCloud, cloud_background,
					firstCloud,	firstCloudFeatures, faceFeatures, finalICP;

	pcl::PointIndices::Ptr largestIndices, largestIndices2, inliers,
													faceIndices, faceIndices2;
	//Get Plane Model
	pcl::ModelCoefficients::Ptr coefficients;
	// Prism object.
	pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism, prism2;

	/*Global descriptor objects*/
	PointCloudNPtr normals;
 	NormalEstimation normalEstimation;
 	pcl::PCDWriter writer;

	PointCloudTVFH308Ptr ourcvfh_desc;
	std::vector<pcl::Vertices> hullPolygons;

	Eigen::Vector4d headHeaight, bgdCentroid;
	Eigen::Vector3d headOrientation;
	Eigen::Matrix4d transformation_matrix;
	double x, y, z, \
	x_sq, y_sq, z_sq, \
	roll, pitch, yaw;  //used to compute spherical coordinates
	double rad_dist, azimuth, polar; //spherical coords

	ros::Publisher posePublisher;
	uint64_t counter;
	std::vector<double> xfilt, yfilt,
						zfilt, pitchfilt,
						yawfilt;

	boost::asio::io_service io_service;
	const std::string multicast_address;
	geometry_msgs::Pose pose_;  // pose of face
	  pcl::IterativeClosestPoint<PointT, PointT> icp;
	  pcl::registration::TransformationEstimationSVD<PointT, PointT, double>::Ptr trans_est_ptr;
public:
	Segmentation(bool running_, ros::NodeHandle nh, bool print)
	: nh_(nh), updateCloud(false), save(false), print(print), running(running_),
	send(false), cloudName("Segmentation Cloud"), savepcd(false), firstFace(true),
	hardware_concurrency(std::thread::hardware_concurrency()), distThreshold(0.7145),
	zmin(0.06f), zmax(0.3553f), v1(0), v2(0), v3(0), v4(0), iterations(2),  spinner(hardware_concurrency/2),
	counter(0),	multicast_address("235.255.0.1")
	{
	    cloud_sub_ = nh.subscribe("ensenso/cloud", 10, &Segmentation::cloudCallback, this);
	}

	~Segmentation()
	{

	}

	void spawn()
	{
		begin();
		end();
	}

	void initPointers()
	{
		//global clouds for class
		segCloud 			= PointCloudTPtr (new PointCloudT);
		plane 				= PointCloudTPtr (new PointCloudT);
		convexHull 			= PointCloudTPtr (new PointCloudT);
		faces 				= PointCloudTPtr (new PointCloudT);
		filteredCloud 		= PointCloudTPtr (new PointCloudT);
		cloud_background 	= PointCloudTPtr (new PointCloudT);
		pillows				= PointCloudTPtr (new PointCloudT);
		firstCloud			= PointCloudTPtr (new PointCloudT);
		firstFaceCloud		= PointCloudTPtr (new PointCloudT);
		firstCloudFeatures	= PointCloudTPtr (new PointCloudT);
		faceFeatures		= PointCloudTPtr (new PointCloudT);
		facesOnly			= PointCloudTPtr (new PointCloudT);
		facesOnly2			= PointCloudTPtr (new PointCloudT);
		passThruCloud 		= PointCloudTPtr (new PointCloudT);
		outlierRemCloud		= PointCloudTPtr (new PointCloudT);
		largest_cluster 	= PointCloudTPtr (new PointCloudT);
		finalICP			= PointCloudTPtr (new PointCloudT);

		//Segmentation Models
		coefficients 		= pcl::ModelCoefficients::Ptr  (new pcl::ModelCoefficients);
		inliers 			= pcl::PointIndices::Ptr  (new pcl::PointIndices);
		faceIndices 		= pcl::PointIndices::Ptr (new pcl::PointIndices);
		faceIndices2 		= pcl::PointIndices::Ptr (new pcl::PointIndices);
		largestIndices		= pcl::PointIndices::Ptr (new pcl::PointIndices);		//indices of segmented face
		largestIndices2 	= pcl::PointIndices::Ptr (new pcl::PointIndices);		//indices of segmented face

		normals 			= PointCloudNPtr (new PointCloudN);

		//descriptors
		ourcvfh_desc 		= PointCloudTVFH308Ptr (new pcl::PointCloud<pcl::VFHSignature308>);

		trans_est_ptr 		= boost::shared_ptr<pcl::registration::TransformationEstimationSVD<PointT, PointT, double>> \
								(boost::make_shared<pcl::registration::TransformationEstimationSVD<PointT, PointT, double>> ());
	}

	void begin()
	{
		if(spinner.canStart())
		{
			spinner.start();
			ROS_INFO("spinning with %lu threads", hardware_concurrency/2);
		}

		while(!updateCloud)
		{
		  std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		posePublisher = nh_.advertise<geometry_msgs::Pose>("/mannequine_head/pose", 1000); // publisher of head pose information

		// Objects for storing the point clouds.
		initPointers();
		//spawn the threads
	    threads.push_back(std::thread(&Segmentation::planeSeg, this));
	    std::for_each(threads.begin(), threads.end(), \
	                  std::mem_fn(&std::thread::join));
	}

	void end()
	{
	  spinner.stop();
	  running = false;
	}

	void getMultiViewer()
	{
		multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));
		multiViewer->initCameraParameters ();
		//top-left
		multiViewer->createViewPort(0.0, 0.5, 0.5, 1.0, v1);
		multiViewer->setBackgroundColor (0, 0, 0, v1);
		multiViewer->addText("Downsampled Cloud", 10, 10, "v1 text", v1);
		//top right
		multiViewer->createViewPort(0.5, 0.5, 1.0, 1.0, v2);
		multiViewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
		multiViewer->addText("Possible faces", 10, 10, "v2 text", v2);
		//bottom-left
		multiViewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
		multiViewer->setBackgroundColor (0.2, 0.2, 0.3, v3);
		multiViewer->addText("EC Segmented Face", 10, 10, "v3 text", v3);
		//bottom-right
		multiViewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);
		multiViewer->setBackgroundColor (0.2, 0.3, 0.2, v4);
		multiViewer->addText("EC Segmented Face", 10, 10, "v4 text", v4);

		multiViewer->registerKeyboardCallback(&Segmentation::keyboardEventOccurred, *this);
	}

	void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
	                            void* )
	{
	  if (event.keyUp())
	  {
	  	switch(event.getKeyCode())
	  	{
	  		case 'p':
	  			savepcd=true;
	  		  break;
	  	}
	  }
	  else if (event.getKeySym () == "space" && event.keyDown ())
	    next_iteration = true;
	}

	void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& ensensoCloud)
	{
		PointCloudT cloud;
		getCloud(ensensoCloud, cloud);
		std::lock_guard<std::mutex> lock(mutex);
		this->cloud = cloud;
		updateCloud = true;
	}

	void getCloud(const sensor_msgs::PointCloud2ConstPtr cb_cloud, PointCloudT& pcl_cloud) const
	{
		pcl::PCLPointCloud2 pcl_pc;
		pcl_conversions::toPCL(*cb_cloud, pcl_pc);
		pcl::fromPCLPointCloud2(pcl_pc, pcl_cloud);
	}


	void saveAllClouds(const PointCloudTPtr segCloud,
					   const PointCloudTPtr faces,
					   const PointCloudTPtr passThroughCloud,
					   const PointCloudTPtr pillows)
	{
		oss.str("");
		oss << counter;
		const std::string baseName = oss.str();

		const std::string origCloud_id = "orig_" + baseName + "_cloud.pcd";// baseName + "_cloud.pcd";
		const std::string facesCloud_id = "segFace_" + baseName + "_cloud.pcd";
		const std::string filteredCloud_id = "filteredSegFace_" + baseName + "_cloud.pcd";
		const std::string backGroundCloud_id = "backCloud_" + baseName + "_cloud.pcd";

		ROS_INFO_STREAM("saving cloud: " << origCloud_id);
		writer.writeBinary(origCloud_id, *segCloud);
		ROS_INFO_STREAM("saving cloud: " << facesCloud_id);
		writer.writeBinary(facesCloud_id, *faces);
		ROS_INFO_STREAM("saving cloud: " << filteredCloud_id);
		writer.writeBinary(filteredCloud_id, *passThroughCloud);
		ROS_INFO_STREAM("saving cloud: " << backGroundCloud_id);
		writer.writeBinary(backGroundCloud_id, *pillows);

		generic::savefaces(faces);

		ROS_INFO_STREAM("saving complete!");
		++counter;
	}

	void getHeadPose(const PointCloudTPtr headNow,
								const PointCloudTPtr filteredSegCloud)
	{
		//first compute the centroid of the retrieved cluster
		//The last compononent of the vector is set to 1, this allows to transform the centroid vector with 4x4 matrices
		Eigen::Vector4d headCentroid, headHeight;
		compute3DCentroid (*headNow, headCentroid);
		//pick position of head at rest
		if(firstFace)
		{
			bgdCentroid = headCentroid;
			firstCloud  = filteredSegCloud;
			firstFaceCloud	= headNow;
		}


		//now subtract the height of camera above table from computed centroid
		headHeight = bgdCentroid - headCentroid;
		firstFace = false;
		headHeight(3) = 1;  		//to allow for rotation

		if(save)
			generic::savepoints(headCentroid);
		/*
		Parametrize a line defined by an origin point o and a unit direction vector d
		$ such that the line corresponds to the set $ l(t) = o + t * d, $ t \in \mathbf{R} $.
		*/
		Eigen::Vector3d back, projx;
		// Eigen::ParametrizedLine<double,4> centralLine;
		back 		<< bgdCentroid(0), bgdCentroid(1), bgdCentroid(2);
		auto faceCenter = headHeight.block<3,1>(0,0);
		/*Compute the projection of the head centroid onto the line*/
		projx  	= faceCenter;
		// projx(0) += 3;  //add a second vector normal to background centroid vexctor
		Eigen::ParametrizedLine<double,3> faceAlongBgd = Eigen::ParametrizedLine<double,\
														3>::Through(faceCenter, back);
		//this is the projection of the face to the x-y plane
		auto projectedFace = faceAlongBgd.projection(faceCenter);

		// Eigen::AlignedVector3<double> faceCenter, bgdCentroid;

 		x    = std::fabs(faceCenter(0)*1000);
 		y 	 = std::fabs(faceCenter(1)*1000);
 		z 	 = std::fabs(faceCenter(2)*1000);

 		x_sq = std::pow(back(0)*1000, 2);
 		y_sq = std::pow(back(1)*1000, 2);
 		z_sq = std::pow(back(2)*1000, 2);

 		//azimuth = yaw rotation about z axis in the counterclockwise direction
 		//polar   = pitch rotation about reference plane in the counterclockwise direction
 		//note that yaw rotation has a secondary effect on roll, the bank angle
		rad_dist 	= std::sqrt(x_sq + y_sq + z_sq);
		polar 	  	= M_PI/2 - std::acos(z/rad_dist);
		//angle between y and projected face will be yaw
		azimuth    	= M_PI/2 - std::atan2(x, projectedFace(1));
		roll 		= std::atan2(x, z);

		// ensenso::HeadPose pose;
		// pose.stamp = ros::Time::now();
		// pose.seq   = ++counter;
		//convert from meters to mm
		headHeight*=1000;

		pose_.position.x = x; //std::fabs(x);
		pose_.position.y = y; //std::fabs(y);
		pose_.position.z = z; //std::fabs(z);
		pose_.orientation.x = azimuth;
		pose_.orientation.y = roll; //M_PI + polar;
		pose_.orientation.z = polar;  //negate this to make pitch +ve
		//convert the angles to degrees
		generic::rad2deg(pose_.position.z);
		generic::rad2deg(pose_.orientation.x);
		generic::rad2deg(pose_.orientation.y);

		pose_.orientation.x -= 35;
		pose_.orientation.y *= 10;

		posePublisher.publish(pose_);
	}

	void pp_callback(const pcl::visualization::PointPickingEvent& event, void*)
	{
	   std::cout << "Picking event active" << std::endl;
	   if(event.getPointIndex()!=-1)
	   {
	       float x,y,z;
	       event.getPoint(x,y,z);
	       std::cout << x<< "; " << y<<"; " << z << std::endl;
	   }
	}


	void passThrough(const PointCloudT& cloud, PointCloudTPtr passThruCloud,
					std::string&& field_name, double&& min_limits, double&& max_limits) const
	{
		//create the filter object and assign it to cloud
		pcl::PassThrough<PointT> filter;
		PointCloudTPtr temp_cloud (new PointCloudT (cloud));
		filter.setInputCloud(temp_cloud);

		filter.setFilterFieldName(field_name);
		filter.setFilterLimits(min_limits, max_limits);
		filter.filter(*passThruCloud);
	}

	bool getLargestCluster(const PointCloudTPtr cloud,
							const pcl::PointIndices::Ptr indices,
							const double &tolerance = 0.032,
							const int &min_size = 100,
							const int &max_size = 850)
	{
	    // Creating the KdTree object for the search method of the extraction
	    TreeKdPtr tree (new TreeKd ());
	    tree->setInputCloud (cloud);

	    //Runs the cluster extraction algorithm
	    std::vector<pcl::PointIndices> cluster_indices;
	    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	    ec.setClusterTolerance (tolerance);
	    ec.setMinClusterSize (min_size);
	    ec.setMaxClusterSize (max_size);
	    ec.setSearchMethod (tree);
	    ec.setInputCloud (cloud);
	    if(indices->indices.size()) ec.setIndices(indices);
	    ec.extract (cluster_indices);

	    // If no clusters were found, return an empty cloud
	    if(!cluster_indices.size()) return false;

	    //Find the largest cluster index
	    auto maxClusterIdx = 0;
	    auto maxCluster = 0;
	    for (int cit =0; cit < cluster_indices.size(); ++cit)
	    {
	        if (cluster_indices[cit].indices.size() > maxCluster)
	        {
	            maxCluster = cluster_indices[cit].indices.size();
	            maxClusterIdx = cit;
	        }
	    }
	    *indices= cluster_indices[maxClusterIdx];
	    return true;
	}

	void planeSeg();
	void getDescriptors();
};

class objectPoseEstim{
private:
	std::mutex mutex_o;
	PointCloudTPtr faces_o;
public:
	objectPoseEstim()
	{

	}

	//destructor
	~objectPoseEstim()
	{

	}

	void initPointers()
	{
		faces_o = PointCloudTPtr(new PointCloudT);
	}

	void computeNormals(const PointCloudTPtr cloud, const PointCloudNPtr cloud_normals, float radius=0.03)
	{
	  // Create the normal estimation class, and pass the input dataset to it
	  NormalEstimation ne;
	  ne.setInputCloud (cloud);
	  // Use all neighbors in a sphere of radius 3cm
	  ne.setRadiusSearch (radius);
	  // Create an empty kdtree representation, and pass it to the normal estimation object.
	  TreeKdPtr localTree (new TreeKd ());
	  ne.setSearchMethod (localTree);
	  // Compute the features
	  ne.compute (*cloud_normals);
	  // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
	}

	// send the clustered cloud to this function
	void computeOURCVFH(const PointCloudTPtr cloud,
						const PointCloudTVFH308Ptr ourcvfh_desc,
						float angle=0.13f, float threshold = 0.025f,
						float axis_ratio = 0.8)
	{
		//get cloud normals
		PointCloudNPtr normals (new PointCloudN);
		computeNormals(cloud, normals);

		//create a our-cvfh object
		pcl::OURCVFHEstimation<PointT, PointN, VFH308> ourcvfh;
		ourcvfh.setInputCloud(cloud);
		ourcvfh.setInputNormals(normals);
		//create an empty kdtree rep to be passed to the fpfh estimation object
		TreeKdPtr tree (new TreeKd());
		ourcvfh.setSearchMethod(tree);
		ourcvfh.setEPSAngleThreshold(angle/*/180.0 * M_PI*/);
		//normalize the bins of the resulting hostogram using total number of points
		ourcvfh.setNormalizeBins(false);
		//normalize the surface descriptors with max size between centroid and cluster points
		ourcvfh.setCurvatureThreshold(threshold);
		ourcvfh.setAxisRatio(axis_ratio);

		ourcvfh.compute(*ourcvfh_desc);

		//now get the transform aligning the cloud to the corresponding SGURF.
		std::vector< Eigen::Matrix4f, Eigen::aligned_allocator< Eigen::Matrix4f > > trans;
		ourcvfh.getTransforms(trans);
		for(auto elem : trans)
			std::cout << "transformations: " << elem << ", ";
		std::cout << std::endl;
	}
};

void Segmentation::getDescriptors()
{
	objectPoseEstim obj;

	float&& tree_rad = 0.03;
	float&& neigh_rad = 0.05;

	// base cloud from which to segment the face
	// PointCloudTPtr cloud (new PointCloudT);
	// {
	// 	std::lock_guard<std::mutex> lock(mutex);
	// 	*cloud = this->cloud;
	// }
	// clustered cloud which contains face
	PointCloudTPtr face_cloud (new PointCloudT);
	{
		std::lock_guard<std::mutex> lock(mutex);
		face_cloud = this->passThruCloud;
	}

	PointCloudTVFH308Ptr ourcvfh_desc (new pcl::PointCloud<pcl::VFHSignature308>);
	float angle, threshold, axis_ratio;
	angle = 5.0; threshold = 1.0; axis_ratio = 0.8;
	obj.computeOURCVFH(face_cloud, ourcvfh_desc, angle, threshold, axis_ratio);
	this->ourcvfh_desc = ourcvfh_desc;
}

void Segmentation::planeSeg()
{	/*Generic initializations*/
	//viewer for segmentation and stuff
	// boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));
	getMultiViewer();
	multiViewer->registerPointPickingCallback(&Segmentation::pp_callback, *this);
	// Create the segmentation object
	pcl::SACSegmentation<PointT> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	/*Set the maximum number of iterations before giving up*/
	seg.setMaxIterations(60);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	/*set the distance to the model threshold to use*/
	// Get the plane model, if present.
	seg.setDistanceThreshold (distThreshold);	//as measured
	{
		std::lock_guard<std::mutex> lock(mutex);
		//it is important to pass this->cloud by ref in order to see updates on screen
		segCloud = boost::shared_ptr<pcl::PointCloud<PointT> >(&this->cloud);
		updateCloud = false;
	}

	/*
	To ease computation, I downsample the input cloud with a
	voxel grid of leaf size 1cm
	*/
	PointCloudTPtr filteredSegCloud(new PointCloudT);
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(segCloud);
	vg.setLeafSize (0.01f, 0.01f, 0.01f);
	vg.filter (*filteredCloud);

	// seg.setInputCloud (filteredSegCloud);
	seg.setInputCloud (segCloud);
	seg.segment (*inliers, *coefficients);

	ROS_INFO_STREAM("segcloud points count: " << segCloud->points.size());
	// ROS_INFO_STREAM("filteredSegCloud points count: " << inliers->points.size());

	// Copy the points of the plane to a new cloud.
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(filteredCloud);
	// extract.setInputCloud(segCloud);
	extract.setIndices(inliers);
	// Get the points associated with the planar surface
	extract.filter(*plane);
	//retrieve convex hull
	pcl::ConvexHull<PointT> hull;
	hull.setInputCloud(plane);

	//ensure hull is 2-d
	hull.setDimension(2);
	hull.reconstruct(*convexHull);

	//redundant check
	if (hull.getDimension() == 2)
	{
		prism.setInputCloud(filteredCloud);
		// prism.setInputCloud(segCloud);
		prism.setInputPlanarHull(convexHull);
		// First parameter: minimum Z value. Set to 0, segments faces lying on the plane (can be negative).
		// Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the faces you expect.
		prism.setHeightLimits(zmin, zmax);
		prism.segment(*faceIndices);

		// Get and show all points retrieved by the hull.
		extract.setIndices(faceIndices);
		extract.filter(*faces);

		getLargestCluster(faces, largestIndices);

		extract.setIndices(largestIndices);
		extract.filter(*facesOnly);

		passThrough(*facesOnly, passThruCloud, std::move("x"),
						std::move(0), std::move(-22));

		// passThrough(*passThruCloud, passThruCloud, std::move("y"),
		//  			std::move(0), std::move(1.5));

		getHeadPose(facesOnly, filteredSegCloud);

		multiViewer->addPointCloud(filteredCloud, "Filtered Cloud", v1);
		// multiViewer->addPointCloud(segCloud, "Filtered Cloud", v1);
		multiViewer->addPointCloud(faces, "Possible faces", v2);
		multiViewer->addPointCloud(facesOnly, "EC Segmented Face", v3);
		multiViewer->addPointCloud(passThruCloud, "PassThru Cloud", v4);

		// multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
		multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Filtered Cloud");
		multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Possible faces");
		multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "EC Segmented Face");
		multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "PassThru Cloud");

	  //   Eigen::Matrix4d trans_est_matrix;
	  //   trans_est_ptr->estimateRigidTransformation(*firstFaceCloud, *facesOnly, trans_est_matrix);
		// Eigen::Affine3d transaff(trans_est_matrix);
  	//     pcl::getTranslationAndEulerAngles(transaff, x, y, z, roll, pitch, yaw);
		// ROS_INFO("SVD Transformation Pose Info: [%4f, %.4f, %.4f, %4f, %.4f, %.4f]: ", headPose.x, headPose.y, headPose.z, roll, pitch, yaw);
		//
		// Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
		while (running && ros::ok())
		{
			/*populate the cloud viewer and prepare for publishing*/
			if(updateCloud)
			{
			  std::lock_guard<std::mutex> lock(mutex);
			  updateCloud = false;

			  vg.setInputCloud(segCloud);
			  vg.filter (*filteredCloud);

			  prism.setInputCloud(filteredCloud);
			  prism.setInputPlanarHull(convexHull);

			  prism.setHeightLimits(zmin, zmax);
			  prism.segment(*faceIndices);
			  // Get and show all points retrieved by the hull.
			  extract.setIndices(faceIndices);
			  extract.filter(*faces);

			  getLargestCluster(faces, largestIndices);
			  extract.setIndices(largestIndices);
			  extract.filter(*facesOnly);

			  passThrough(*facesOnly, passThruCloud, std::move("x"),
			  				std::move(0), std::move(2));

			  getHeadPose(facesOnly, filteredSegCloud);

			  //broadcast the pose to the network
			  if(send)
			    udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), pose_);

  	    	  multiViewer->updatePointCloud(filteredCloud, "Downsampled Cloud");
  	    	  multiViewer->updatePointCloud(faces, "Possible faces");
  	    	  multiViewer->updatePointCloud(facesOnly, "EC Segmented Face");
  	    	  multiViewer->updatePointCloud(passThruCloud, "PassThru Cloud");

			  // compute our-cvfh 6-DOF pose
			  this->getDescriptors();

			  if(print)
			  	ROS_INFO("# pts in faces cloud: %lu | passthru cloud: %lu | EC cloud: %lu", faces->points.size(),
			  		passThruCloud->points.size(), facesOnly->points.size());
			}
			if(savepcd)
			{
				savepcd = false;
				saveAllClouds(segCloud, faces, facesOnly, outlierRemCloud);
			}
			multiViewer->spinOnce(1);
		}
		multiViewer->close();
	}
	else
		ROS_INFO("Chosen hull is not planar");
}

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "ensensor_segmentation_node");
	ROS_INFO("Started node %s", ros::this_node::getName().c_str());

	bool running, print, disp;
	print = false;
	running = true;
	disp = false;

	if (pcl::console::find_argument (argc, argv, "-p") >= 0)
	{
	  print = true;
	}

	ros::NodeHandle nh;

	Segmentation seg(running, nh, print);
	seg.spawn();

	ros::shutdown();
}
