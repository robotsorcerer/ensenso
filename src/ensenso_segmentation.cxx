#include <mutex>
#include <cmath>
#include <thread>
#include <memory>
#include <fstream>
#include <ros/spinner.h>
#include <sensor_msgs/PointCloud2.h>

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
#include <pcl/segmentation/extract_polygonal_prism_data.h>

/*Computer Descriptors*/
#include <pcl/common/centroid.h>
#include <pcl/console/parse.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>

class objectPoseEstim; //class forward declaration
class sender; //class forward declaration of boost broadcaster

class Segmentation
{
private:	
	friend class objectPoseEstim; //friend class forward declaration
	bool updateCloud, save, running, print, savepcd, send;
	ros::NodeHandle nh_;
	std::ostringstream oss;
 	const std::string cloudName;
	ros::Subscriber cloud_sub_;
	std::mutex mutex;
 	PointCloudT cloud;  
 	std::thread cloudDispThread, normalsDispThread;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, segViewer;

	boost::shared_ptr<visualizer> viz;
	std::vector<std::thread> threads;	
 	unsigned long const hardware_concurrency;
	float distThreshold; //distance at which to apply the threshold
	float zmin, zmax;	//minimum and maximum distance to extrude the prism object
	int v1, v2, v3, v4;
 	ros::AsyncSpinner spinner;
 	/*Filtered Cloud and backgrounds*/
 	PointCloudTPtr filteredCloud, cloud_background;
 	mutable PointCloudTPtr pillows;   //cause we'll copy indices of pillows to this
	boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer;

	/*Plane Segmentation Objects*/
	PointCloudTPtr segCloud, plane, convexHull, faces;
	//Get Plane Model
	pcl::ModelCoefficients::Ptr coefficients;
	//indices of segmented face
	pcl::PointIndices::Ptr inliers, faceIndices;
	// Prism object.
	pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism, prism2;

	/*Global descriptor objects*/
	PointCloudNPtr normals;
 	NormalEstimation normalEstimation;
 	pcl::PCDWriter writer;

	PointCloudTVFH308Ptr ourcvfh_desc;
	PointCloudTVFH308Ptr vfh_desc;
	PointCloudPFH125Ptr pfh_desc;
	PointCloudCRH90Ptr crhHistogram;
  	std::vector<pcl::Vertices> hullPolygons;

	Eigen::Vector4d headHeaight, bgdCentroid;
	Eigen::Vector3d headOrientation;
	ensenso::HeadPose headPose;

	PointCloudTPtr facesOnly, passThruCloud, 
					largest_cluster, outlierRemCloud;
	pcl::PointIndices::Ptr largestIndices;		

	double x, y, z, \
	x_sq, y_sq, z_sq;  //used to compute spherical coordinates
	
	
	double rad_dist, azimuth, polar; //spherical coords

	ros::Publisher posePublisher;
	uint64_t counter;
	std::vector<double> xfilt, yfilt, 
						zfilt, pitchfilt,
						yawfilt;

	boost::asio::io_service io_service;
	const std::string multicast_address;
public:
	Segmentation(bool running_, ros::NodeHandle nh, bool print)
	: nh_(nh), updateCloud(false), save(false), print(print), running(running_), 
	send(false), cloudName("Segmentation Cloud"), savepcd(false),
	hardware_concurrency(std::thread::hardware_concurrency()), distThreshold(0.712), 
	zmin(0.2f), zmax(0.4953f), v1(0), v2(0), v3(0), v4(0), spinner(hardware_concurrency/2), counter(0),
	multicast_address("235.255.0.1")
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
		facesOnly			= PointCloudTPtr (new PointCloudT);
		passThruCloud 		= PointCloudTPtr (new PointCloudT);
		outlierRemCloud		= PointCloudTPtr (new PointCloudT);
		largest_cluster 	= PointCloudTPtr  (new PointCloudT);

		//Segmentation Models
		coefficients 		= pcl::ModelCoefficients::Ptr  (new pcl::ModelCoefficients);
		inliers 			= pcl::PointIndices::Ptr  (new pcl::PointIndices);
		faceIndices 		= pcl::PointIndices::Ptr (new pcl::PointIndices);
		largestIndices		= pcl::PointIndices::Ptr (new pcl::PointIndices);		//indices of segmented face

		normals 			= PointCloudNPtr (new PointCloudN);	
		// multiViewer 		= boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));

		//descriptors
		ourcvfh_desc 		= PointCloudTVFH308Ptr (new pcl::PointCloud<pcl::VFHSignature308>);
		vfh_desc 			= PointCloudTVFH308Ptr (new pcl::PointCloud<pcl::VFHSignature308>);
		pfh_desc 			= PointCloudPFH125Ptr(new PointCloudPFH125);
		crhHistogram 		= PointCloudCRH90Ptr (new PointCloudCRH90);
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

		// Objects for storing the point clouds.
		initPointers();
		bgdCentroid << 0.0, 0.0, distThreshold, 1.0;
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

	void getMultiViewer(/*boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer*/)
	{				
		multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));
		multiViewer->initCameraParameters ();
		//top-left
		multiViewer->createViewPort(0.0, 0.5, 0.5, 1.0, v1);
		multiViewer->setBackgroundColor (0, 0, 0, v1);
		multiViewer->addText("Original Cloud", 10, 10, "v1 text", v1);
		//top right
		multiViewer->createViewPort(0.5, 0.5, 1.0, 1.0, v2);
		multiViewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
		multiViewer->addText("Downsampled Cloud", 10, 10, "v2 text", v2);
		//bottom-left
		multiViewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
		multiViewer->setBackgroundColor (0.2, 0.2, 0.3, v3);
		multiViewer->addText("Possible faces", 10, 10, "v3 text", v3);
		//bottom-right
		multiViewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);
		multiViewer->setBackgroundColor (0.2, 0.3, 0.2, v4);
		multiViewer->addText("Segmented Face", 10, 10, "v4 text", v4);

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

	ensenso::HeadPose getHeadPose(const PointCloudTPtr headNow)
	{
		//first compute the centroid of the retrieved cluster
		//The last compononent of the vector is set to 1, this allows to transform the centroid vector with 4x4 matrices
		Eigen::Vector4d headCentroid, headHeight;
		compute3DCentroid (*headNow, headCentroid);
		//now subtract the height of camera above table from computed centroid
		headHeight = bgdCentroid - headCentroid;
		headHeight(3) = 1;  		//to allow for rotation

		if(save)
			generic::savepoints(headCentroid);

		//compute the orientation of the segmented cluster about the background
		Eigen::Vector3d headAngle;

	    if(print)
	    {
		    ROS_INFO("HeadHeight [x: %f, y: %f, z: %f ]", headHeight(0), headHeight(1), headHeight(2) );
		    ROS_INFO("HeadCentroids [x: %f, y: %f, z: %f ]", headCentroid(0), headCentroid(1), headCentroid(2) );
	    }

		/*
		Parametrize a line defined by an origin point o and a unit direction vector d
		$ such that the line corresponds to the set $ l(t) = o + t * d, $ t \in \mathbf{R} $.
		*/		
		Eigen::Vector3d back, projback;
		// Eigen::ParametrizedLine<double,4> centralLine;
		back 		<< bgdCentroid(0), bgdCentroid(1), bgdCentroid(2);
		projback  	= back;
		projback(2) += 1;  //add a second vector normal to background centroid vexctor
		Eigen::ParametrizedLine<double,3> centralLine = Eigen::ParametrizedLine<double,\
														3>::Through(back, projback);
		/*Compute the projection of the head centroid onto the line*/
		auto faceCenter = headHeight.block<3,1>(0,0);
		auto projectedFace = centralLine.projection(faceCenter);	
		if(print){			
			ROS_INFO_STREAM("projected line: \n" << projectedFace);	
			ROS_INFO_STREAM("head center: \n" << faceCenter);	
			// ROS_INFO_STREAM("parametereized line: " << centralLine);	
		}											
		/*
 		compute the spherical coordinates of the face cluster with origin as the background cluster of the base pillow
 		I use the ISO convention with my azimuth being the signed angle measured from azimuth reference direction to the orthogonal projection of 
 		line segment OP on the reference plane 
		*/
 		// r = sqrt(x^2 + y^2 + z^2)
 		x    = faceCenter(0) /*- projectedFace(0)*/;
 		y 	 = faceCenter(1) /*- projectedFace(1)*/;
 		z 	 = faceCenter(2) /*- projectedFace(2)*/;

 		x_sq = std::pow(x, 2);
 		y_sq = std::pow(y, 2);
 		z_sq = std::pow(z, 2);

 		//azimuth = yaw rotation about z axis in the counterclockwise direction
 		//polar   = pitch rotation about reference plane in the counterclockwise direction
 		//note that yaw rotation has a secondary effect on roll, the bank angle
		rad_dist = std::sqrt(x_sq + y_sq + z_sq);
		azimuth  = std::acos(z/rad_dist);
		polar    = std::atan(y/x);

		ensenso::HeadPose pose;
		pose.stamp = ros::Time::now();
		pose.seq   = ++counter;
		//convert from meters to mm
		headHeight*=1000;
/*		//filter the 5-dof pose with my savgol impl
		// xfilt, yfilt, zfilt, polarfilt, yawfilt
		xfilt.push_back(headHeight(0));
		yfilt.push_back(headHeight(1));
		zfilt.push_back(headHeight(2));
		pitchfilt.push_back(generic::rad2deg(polar));
		yawfilt.push_back(generic::rad2deg(azimuth));

		if(counter%5==0)
		{
			savgolfilt(xfilt, 3, 5);
			savgolfilt(yfilt, 3, 5);
			savgolfilt(zfilt, 3, 5);
			savgolfilt(pitchfilt, 3, 5);
			savgolfilt(yawfilt, 3, 5);
		}*/

		pose.x = headHeight(0);
		pose.y = headHeight(1);
		pose.z = headHeight(2);
		pose.pitch = polar;
		pose.yaw = azimuth;
		//convert the angles to degrees
		generic::rad2deg(pose.pitch);
		generic::rad2deg(pose.yaw);
		//negate x for convenience
		// pose.x = -pose.x;

		posePublisher = nh_.advertise<ensenso::HeadPose>("/mannequine_head/pose", 1000);
		posePublisher.publish(pose);

		return pose;
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

	bool getLargestCluster(const PointCloudTPtr cloud, 
							const pcl::PointIndices::Ptr indices, 
							const double &tolerance = 0.052, 
							const int &min_size = 300, 
							const int &max_size = 2500)
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

	void planeSeg()
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
		vg.filter (*filteredSegCloud);

		seg.setInputCloud (filteredSegCloud);
		seg.segment (*inliers, *coefficients);

		// Copy the points of the plane to a new cloud.
		pcl::ExtractIndices<PointT> extract;
		extract.setInputCloud(filteredSegCloud);
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
			prism.setInputCloud(filteredSegCloud);
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

			headPose = getHeadPose(facesOnly);

			multiViewer->addPointCloud(segCloud, "Original Cloud", v1);
			multiViewer->addPointCloud(filteredSegCloud, "Downsampled Cloud", v2);
			multiViewer->addPointCloud(faces, "Possible faces", v3);
			multiViewer->addPointCloud(facesOnly, "Segmented Face", v4);

			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Downsampled Cloud");
			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Possible faces");
			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Face");

			while (running && ros::ok())
			{
				/*populate the cloud viewer and prepare for publishing*/
				if(updateCloud)
				{   
				  std::lock_guard<std::mutex> lock(mutex); 
				  updateCloud = false;
				  multiViewer->updatePointCloud(segCloud, "Original Cloud");

				  vg.setInputCloud(segCloud);
				  vg.filter (*filteredSegCloud);

				  prism.setInputCloud(filteredSegCloud);
				  prism.setInputPlanarHull(convexHull);

				  prism.setHeightLimits(zmin, zmax);
				  prism.segment(*faceIndices);
				  // Get and show all points retrieved by the hull.
				  extract.setIndices(faceIndices);
				  extract.filter(*faces);				  
				  multiViewer->updatePointCloud(filteredSegCloud, "Downsampled Cloud");

				  getLargestCluster(faces, largestIndices);
				  extract.setIndices(largestIndices);
				  extract.filter(*facesOnly);

				  multiViewer->updatePointCloud(faces, "Possible faces");
				  multiViewer->updatePointCloud(facesOnly, "Segmented Face");

				  if(print){
				  ROS_INFO("orig cloud has %lu points", segCloud->points.size());
				  ROS_INFO("downsampled face has %lu points", faces->points.size());				  	
				  }

				  headPose = getHeadPose(facesOnly);
				  //broadcast the pose to the network
				  if(send)
				    udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), headPose);
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

	  // Create an empty kdtree representation, and pass it to the normal estimation object.
	  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	  TreeKdPtr localTree (new TreeKd ());
	  ne.setSearchMethod (localTree);

	  // Use all neighbors in a sphere of radius 3cm
	  ne.setRadiusSearch (radius);

	  // Compute the features
	  ne.compute (*cloud_normals);
	  // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
	}

	void computeOURCVFH(const PointCloudTPtr cloud, const PointCloudTVFH308Ptr ourcvfh_desc, float angle=5.0,\
						float threshold = 1.0, float axis_ratio = 0.8)
	{
		//get cloud normals
		PointCloudNPtr normals (new PointCloudN);
		computeNormals(cloud, normals);

		//create a our_cvfh object
		pcl::OURCVFHEstimation<PointT, PointN, VFH308> ourcvfh;
		ourcvfh.setInputCloud(cloud);
		ourcvfh.setInputNormals(normals);
		//create an empty kdtree rep to be passed to the fpfh estimation object
		TreeKdPtr tree (new TreeKd());
		ourcvfh.setSearchMethod(tree);
		ourcvfh.setEPSAngleThreshold(angle/180.0 * M_PI);
		//normalize the bins of the resulting hostogram using total number of points
		ourcvfh.setNormalizeBins(true);
		//normalize the surface descriptors with max size between centroid and cluster points
		ourcvfh.setCurvatureThreshold(threshold);
		ourcvfh.setAxisRatio(axis_ratio);

		ourcvfh.compute(*ourcvfh_desc);
	}

	void computeVFH(const PointCloudTPtr cloud, const PointCloudTVFH308Ptr vfh_desc)
	{
		//get cloud normals
		PointCloudNPtr normals (new PointCloudN);
		computeNormals(cloud, normals);

		//create a vfh object
		pcl::VFHEstimation<PointT, PointN, VFH308> vfh;
		vfh.setInputCloud(cloud);
		vfh.setInputNormals(normals);
		//create an empty kdtree rep to be passed to the fpfh estimation object
		TreeKdPtr tree (new TreeKd());
		vfh.setSearchMethod(tree);
		//normalize the bins of the resulting hostogram using total number of points
		vfh.setNormalizeBins(true);
		//normalize the surface descriptors with max size between centroid and cluster points
		vfh.setNormalizeDistance(false);

		vfh.compute(*vfh_desc);
	}

	void computePFH(const PointCloudTPtr cloud, const PointCloudPFH125Ptr pfh_desc, \
					const float& tree_rad = 0.03, const float& neigh_rad = 0.05)
	{
		PointCloudNPtr normals(new PointCloudN);
		NormalEstimation nest;
		nest.setInputCloud(cloud);
		nest.setRadiusSearch(tree_rad);

		TreeKdPtr kdt (new TreeKd);

		nest.setSearchMethod(kdt);
		nest.compute(*normals);

		pcl::PFHEstimation<PointT, PointN, PFH125> PFH;
		//pfh processing
		PFH.setInputCloud(cloud);
		PFH.setInputNormals(normals);
		PFH.setSearchMethod(kdt);

		if(neigh_rad <= tree_rad)
		{
			std::cerr << "Neighborhood radius must be greater than the normals estimation radius" << std::endl;
		}

		PFH.setRadiusSearch(neigh_rad);

		PFH.compute(*pfh_desc);
	}

	/*This is my vfh descriptor of the cluster I have identified in the scene*/
	void cameraRollHistogram(const PointCloudTPtr faces, const PointCloudCRH90Ptr histogram)
	{	
		//normals for crhobjects
		PointCloudNPtr normals(new PointCloudN); 

		//read the snapshot of the object from Segmentation Class and
		//compute its normals
		computeNormals(faces, normals);

		// CRH estimation object.
		pcl::CRHEstimation<PointT, PointN, CRH90> crh;
		crh.setInputCloud(faces);
		crh.setInputNormals(normals);
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*faces, centroid);
		crh.setCentroid(centroid);

		// Compute the CRH.
		crh.compute(*histogram);
	}
};

void Segmentation::getDescriptors()
{	
	objectPoseEstim obj;	

	float&& tree_rad = 0.03;
	float&& neigh_rad = 0.05;

	PointCloudTPtr cloud (new PointCloudT);
	{		
		std::lock_guard<std::mutex> lock(mutex);
		*cloud = this->cloud;
	}
	PointCloudTVFH308Ptr ourcvfh_desc (new pcl::PointCloud<pcl::VFHSignature308>);
	float angle, threshold, axis_ratio;
	angle = 5.0; threshold = 1.0; axis_ratio = 0.8;
	obj.computeOURCVFH(cloud, ourcvfh_desc, angle, threshold, axis_ratio);
	this->ourcvfh_desc = ourcvfh_desc;

	//compute vfh descriptors
	PointCloudTVFH308Ptr vfh_desc (new pcl::PointCloud<pcl::VFHSignature308>);
	obj.computeVFH(cloud, vfh_desc);
	this->vfh_desc = vfh_desc;

	//get pfh
	PointCloudPFH125Ptr pfh_desc(new PointCloudPFH125);
	obj.computePFH(cloud, pfh_desc, tree_rad, neigh_rad);
	this->pfh_desc = pfh_desc; 

	//compute camera roll histogram along with scene descriptors
	PointCloudCRH90Ptr crhHistogram (new PointCloudCRH90);
	obj.cameraRollHistogram(cloud, crhHistogram);
	this->crhHistogram = crhHistogram;
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