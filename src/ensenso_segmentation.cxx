#include <mutex>
#include <cmath>
#include <thread>
#include <memory>
#include <fstream>
#include <ros/spinner.h>

#include <ensenso/visualizer.h>
#include <ensenso/pcl_headers.h>
#include <ensenso/camera_matrices.h>
#include <ensenso/ensenso_headers.h>
#include <sensor_msgs/PointCloud2.h>
#include <ensenso/HeadPose.h> //local msg communicator of 5-d pose

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

/*Globlal namespaces and aliases*/
using PointT 			= pcl::PointXYZ;
using PointCloudT 		= pcl::PointCloud<PointT>;
using PointCloudTPtr 	= PointCloudT::Ptr;

using PointN  			= pcl::Normal;
using PointCloudN 		= pcl::PointCloud<PointN>;
using PointCloudNPtr  	= PointCloudN::Ptr;

using pcl_viz 			= pcl::visualization::PCLVisualizer;
using NormalEstimation 	= pcl::NormalEstimation<PointT, PointN>;

using CRH90 				= pcl::Histogram<90>;
using PointCloudCRH90		= pcl::PointCloud<CRH90>;
using PointCloudCRH90Ptr	= PointCloudCRH90::Ptr;

/*Descriptor aliases*/
using PFH125 				= pcl::PFHSignature125;
using PointCloudPFH125 		= pcl::PointCloud<PFH125>;
using PointCloudPFH125Ptr 	= pcl::PointCloud<PFH125>::Ptr;

/*Kd Trees*/
using TreeKd = pcl::search::KdTree<PointT>;
using TreeKdPtr = pcl::search::KdTree<PointT>::Ptr;

using VFH308 = pcl::VFHSignature308;
using PointCloudVFH308 = pcl::PointCloud<VFH308>;
using PointCloudTVFH308Ptr = PointCloudVFH308::Ptr;

#define OUT(__o__) std::cout<< __o__ << std::endl;


// void getTransformationMatrix()
// {
//   if(!initCaptureParams)
//   {
//     ROS_WARN("Camera not initialized");
//   }
//   else
//   {
//     std::string jsonTree = ensenso_ptr->getResultAsJson();
//     ROS_INFO_STREAM(jsonTree);
//     ensenso_ptr->initExtrinsicCalibration (14);
//   }

// }

class objectPoseEstim; //class forward declaration

class Segmentation
{
private:	
	friend class objectPoseEstim; //friend class forward declaration
	bool updateCloud, save, running, print, savepcd;
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

	double x, y, z, \
	x_sq, y_sq, z_sq;  //used to compute spherical coordinates
	
	
	double rad_dist, azimuth, polar; //spherical coords

	ros::Publisher posePublisher;
	uint64_t counter;
public:
	Segmentation(bool running_, ros::NodeHandle nh, bool print)
	: nh_(nh), updateCloud(false), save(false), print(print), running(running_), 
	cloudName("Segmentation Cloud"), savepcd(false),
	hardware_concurrency(std::thread::hardware_concurrency()), distThreshold(0.712), zmin(0.2f), zmax(0.4953f),
	v1(0), v2(0), v3(0), v4(0), spinner(hardware_concurrency/2), counter(0)
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

		//Segmentation Models
		coefficients 		= pcl::ModelCoefficients::Ptr  (new pcl::ModelCoefficients);
		inliers 			= pcl::PointIndices::Ptr  (new pcl::PointIndices);
		faceIndices 		= pcl::PointIndices::Ptr (new pcl::PointIndices);		//indices of segmented face

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
		bgdCentroid << 0.0, 0.0, distThreshold, 1.0;//getBackGroundCentroid();	
		//spawn the threads
	    threads.push_back(std::thread(&Segmentation::planeSeg, this));
	    // threads.push_back(std::thread(&Segmentation::cloudDisp, this));
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
		multiViewer->addText("Segmented Cloud", 10, 10, "v2 text", v2);
		//bottom-left
		multiViewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
		multiViewer->setBackgroundColor (0.2, 0.2, 0.3, v3);
		multiViewer->addText("PassThrough Filtered Segmented Cloud", 10, 10, "v3 text", v3);
		//bottom-right
		multiViewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);
		multiViewer->setBackgroundColor (0.2, 0.3, 0.2, v4);
		multiViewer->addText("Base IAB Cluster", 10, 10, "v4 text", v4);

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
		
		savefaces(faces);

		ROS_INFO_STREAM("saving complete!");
		++counter;
	}

	void savefaces(const PointCloudTPtr faces)
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

	void savepoints(Eigen::Vector4d &centroid)
	{
	    //Now we write the points to a text file for visualization processing
	    std::ofstream midface("pose.csv", std::ios_base::app | std::ios_base::out);
	    midface << centroid(0) <<"\t" <<centroid(1)<< "\t" << centroid(2) << "\n";
	    ROS_INFO("Writing %f, %f, %f to pose.csv", centroid(0), centroid(12), centroid(2));
	    midface.close();
	}

	/*Here I used Euclidean clustering to get the clouds of the base pillow which happens to be
	the largest cluster in the scene*/
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
		auto max = clustersVec[0]->points.size();
		pcl::PointCloud<pcl::PointXYZ>::Ptr biggestCluster (new pcl::PointCloud<pcl::PointXYZ>);
		for(auto x : clustersVec)
		{
		  if(max <= x->points.size() ) 
		    {
		      max = x->points.size();
		      biggestCluster = x;
		    }
		}
		std::cout << "biggest Cluster has " << biggestCluster->points.size () << " data points." << std::endl;

		this->pillows = biggestCluster;
	}

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

	double findMax(const PointCloudTPtr cloud_in)
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
/*
 * Gets the orientation quaternion of a mesh model such that the z-axis is normal to the plane, and the x-y
 * axis are as close as possible to the the table frame
 *
 * \param headNow  The point cloud of the head
 *
 * \returns quaternion  The quaternion that would rotate the origin frame to fit the head model (z-normal)
 * */

	ensenso::HeadPose getHeadPose(const PointCloudTPtr headNow)
	{
		//first compute the centroid of the retrieved cluster
		//The last compononent of the vector is set to 1, this allows to transform the centroid vector with 4x4 matrices
		Eigen::Vector4d headCentroid, headHeight;
		compute3DCentroid (*headNow, headCentroid);
		//now subtract the height of camera above table from computed centroid
		headHeight = bgdCentroid - headCentroid;
		ROS_INFO_STREAM("headHeight = " << headHeight);
		headHeight(3) = 1;  		//to allow for rotation

		if(save)
			savepoints(headCentroid);

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
		projback(2) += 1;
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
		pose.x = headHeight(0)*1000;
		pose.y = headHeight(1)*1000;
		pose.z = headHeight(2)*1000;
		pose.pitch = polar;
		pose.yaw = azimuth;

		posePublisher = nh_.advertise<ensenso::HeadPose>("/mannequine_head/pose", 1000);
		posePublisher.publish(pose);

		return pose;
	}

	void cloudDisp() 
	{
	  const PointCloudT& cloud  = this->cloud;    
	  PointCloudT::ConstPtr cloud_ptr;	  
	  // PointCloudT::ConstPtr cloud_ptr(new PointCloudT::ConstPtr);
	  // *cloud_ptr = this->cloud;
	  cloud_ptr = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >(&this->cloud);
	  
	  pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 255, 255);	
	  viz = boost::shared_ptr<visualizer> (new visualizer());

	  viewer= viz->createViewer();    
	  viewer->addPointCloud(cloud_ptr, color_handler, cloudName);

	  for(; running && ros::ok() ;)
	  {
	    //populate the cloud viewer and prepare for publishing
	    if(updateCloud)
	    {   
	      std::lock_guard<std::mutex> lock(mutex); 
	      updateCloud = false;
	      viewer->updatePointCloud(cloud_ptr, cloudName);
	    }	    
	    viewer->spinOnce(10);
	  }
	  viewer->close();
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
		getBackGroundCentroid();  //retrieve pillows
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

			// PointCloudT radius   
			PointCloudTPtr passThruCloud  (new PointCloudT), outlierRemCloud(new PointCloudT);
			passThrough(*faces, passThruCloud);
			outlierRemoval(faces, outlierRemCloud);

			headPose = getHeadPose(faces);

			multiViewer->addPointCloud(segCloud, "Original Cloud", v1);
			multiViewer->addPointCloud(faces, "Segmented Cloud", v2);
			multiViewer->addPointCloud(passThruCloud, "Filtered Cloud", v3);

			multiViewer->addPointCloud(outlierRemCloud, "Base IAB Cluster", v4);

			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Cloud");
			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Filtered Cloud");

			multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Base IAB Cluster");

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
				  multiViewer->updatePointCloud(faces, "Segmented Cloud");

				  //filter segmented faces
				  passThrough(*faces, passThruCloud);
				  multiViewer->updatePointCloud(passThruCloud, "Filtered Cloud");

				  outlierRemoval(faces, outlierRemCloud);
				  multiViewer->updatePointCloud(outlierRemCloud, "Base IAB Cluster");

				  if(print){
				  ROS_INFO("trimmedg cloud has %lu points", passThruCloud->points.size());
				  ROS_INFO("orig cloud has %lu points", segCloud->points.size());
				  ROS_INFO("downsampled face has %lu points", faces->points.size());				  	
				  }

				  headPose = getHeadPose(faces);
				}	 
				if(savepcd)
				{
					savepcd = false;
					saveAllClouds(segCloud, faces, passThruCloud, this->pillows);
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