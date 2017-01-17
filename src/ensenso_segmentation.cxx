#include <mutex>
#include <thread>
#include <ros/spinner.h>

#include <ensenso/visualizer.h>
#include <ensenso/pcl_headers.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

/*Computer Descriptors*/
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/features/crh.h>

#include <pcl/filters/statistical_outlier_removal.h>

/*Globlal namespaces and aliases*/
// using namespace pathfinder;
using PointT 			= pcl::PointXYZ;
using PointCloudT 		= pcl::PointCloud<PointT>;
using PointCloudTPtr 	= PointCloudT::Ptr;

using PointN  			= pcl::Normal;
using PointCloudN 		= pcl::PointCloud<PointN>;
using PointCloudNPtr  	= PointCloudN::Ptr;

using pcl_viz 			= pcl::visualization::PCLVisualizer;
using NormalEstimation 	= pcl::NormalEstimation<PointT, PointN>;

using CRH90 			= pcl::Histogram<90>;
using PointCloudH 		= pcl::PointCloud<CRH90>;
using PointCloudHPtr	= PointCloudH::Ptr;

#define OUT(__o__) std::cout<< __o__ << std::endl;

class Segmentation
{
private:
	bool updateCloud, save, running;
	ros::NodeHandle nh_;
 	const std::string cloudName;
	ros::Subscriber cloud_sub_;
	std::mutex mutex;
 	PointCloudT cloud;  
 	std::thread cloudDispThread, normalsDispThread;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, segViewer;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer;

	boost::shared_ptr<visualizer> viz;
	std::vector<std::thread> threads;	
 	unsigned long const hardware_concurrency;
	float distThreshold; //distance at which to apply the threshold
	float zmin, zmax;	//minimum and maximum distance to extrude the prism object
	int v1, v2, v3, v4;
 	ros::AsyncSpinner spinner;
 	/*Filtered Cloud*/
 	PointCloudTPtr filteredCloud;

	/*Plane Segmentation Objects*/
	PointCloudTPtr segCloud, plane, convexHull, faces;
	//Get Plane Model
	pcl::ModelCoefficients::Ptr coefficients;
	//indices of segmented face
	pcl::PointIndices::Ptr inliers, faceIndices;
	// Prism object.
	pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;

	/*Global descriptor objects*/
	PointCloudNPtr normals;
	PointCloudHPtr histogram;
 	NormalEstimation normalEstimation;
public:
	Segmentation(bool running_, ros::NodeHandle nh)
	: nh_(nh), updateCloud(false), save(false), running(running_), cloudName("Segmentation Cloud"),
	hardware_concurrency(std::thread::hardware_concurrency()), distThreshold(0.712), zmin(0.2f), zmax(0.6953f),
	v1(0), v2(0), v3(0), v4(0), spinner(hardware_concurrency/2)
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
		segCloud 	= pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
		plane 		= 	pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
		convexHull 	= pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
		faces 		= pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
		filteredCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
		//Get Plane Model
		coefficients = pcl::ModelCoefficients::Ptr  (new pcl::ModelCoefficients);
		inliers 	= pcl::PointIndices::Ptr  (new pcl::PointIndices);
		//indices of segmented face
		faceIndices = pcl::PointIndices::Ptr (new pcl::PointIndices);
		normals 	= PointCloudNPtr (new PointCloudN);
		histogram 	= PointCloudHPtr (new PointCloudH);
		multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));  
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
		//spawn the threads
	    threads.push_back(std::thread(&Segmentation::planeSeg, this));
	    // threads.push_back(std::thread(&Segmentation::cloudDisp, this));
	    // threads.push_back(std::thread(&Segmentation::cameraRollHistogram, this));
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
		multiViewer->addText("Convex Hull", 10, 10, "v4 text", v4);
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

		//filter points with particular z range
		filter.setFilterFieldName("x");
		filter.setFilterLimits(0.0, 1000);
		filter.filter(*passThruCloud);
	}

	void meanKFilter(const PointCloudT& cloud, PointCloudTPtr filteredCloud) const
	{
		pcl::StatisticalOutlierRemoval<PointT> filter;
		PointCloudTPtr temp_cloud (new PointCloudT (cloud));
		filter.setInputCloud(temp_cloud);
		//set numbers of neighbors to consider 50
		filter.setMeanK(10);
		/*set standard deviation to multiply with to 1
		points with a distance larger than 1 std. dev of the mean outliers 
		*/
		filter.setStddevMulThresh(15.0);
		filter.filter(*filteredCloud);
	}

	void cloudDisp() 
	{
	  const PointCloudT& cloud  = this->cloud;    
	  PointCloudT::ConstPtr cloud_ptr (&cloud);	  
	  /*we must */
	  // PointCloudT::ConstPtr cloud_ptr(new PointCloudT::ConstPtr);
	  cloud_ptr = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >(&this->cloud);
	  
	  pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 255, 255);	
	  viz = boost::shared_ptr<visualizer> (new visualizer());

	  viewer= viz->createViewer();    
	  viewer->addPointCloud(cloud_ptr, color_handler, cloudName);

	  for(; running && ros::ok() ;)
	  {
	    /*populate the cloud viewer and prepare for publishing*/
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

	void planeSeg()
	{	/*Generic initializations*/
		//viewer for segmentation and stuff 
		getMultiViewer();
		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients (true);
		/*Set the maximum number of iterations before giving up*/
		seg.setMaxIterations(60);
		seg.setModelType (pcl::SACMODEL_PLANE); 
		seg.setMethodType (pcl::SAC_RANSAC);
		/*set the distance to the model threshold to use*/
		seg.setDistanceThreshold (distThreshold);	//as measured
		{
			std::lock_guard<std::mutex> lock(mutex);	
			//it is important to pass this->cloud by ref in order to see updates on screen
			segCloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >(&this->cloud);
			updateCloud = false;	
		}
		seg.setInputCloud (segCloud);
		seg.segment (*inliers, *coefficients);

		if(inliers->indices.size() == 0)
			ROS_INFO("Could not find plane in scene");
		else
		{
			// Copy the points of the plane to a new cloud.
			pcl::ExtractIndices<pcl::PointXYZ> extract;
			extract.setInputCloud(segCloud);
			extract.setIndices(inliers);
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
				prism.setInputCloud(segCloud);
				prism.setInputPlanarHull(convexHull);
				// First parameter: minimum Z value. Set to 0, segments faces lying on the plane (can be negative).
				// Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the faces you expect.
				prism.setHeightLimits(zmin, zmax);
				prism.segment(*faceIndices);

				// Get and show all points retrieved by the hull.
				extract.setIndices(faceIndices);
				extract.filter(*faces);

				//filter segmented faces
				meanKFilter(*faces, filteredCloud);
				// PointCloudT radius   
				PointCloudTPtr passThruCloud  (new PointCloudT);
				passThrough(*faces, passThruCloud);

				multiViewer->addPointCloud(segCloud, "Original Cloud", v1);
				multiViewer->addPointCloud(faces, "Segmented Cloud", v2);
				multiViewer->addPointCloud(passThruCloud, "Filtered Cloud", v3);
				multiViewer->addPointCloud(filteredCloud, "Convex Hull", v4);

				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Filtered Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Convex Hull");

				while (running && ros::ok())
				{
					/*populate the cloud viewer and prepare for publishing*/
					if(updateCloud)
					{   
					  std::lock_guard<std::mutex> lock(mutex); 
					  updateCloud = false;
					  multiViewer->updatePointCloud(segCloud, "Original Cloud");

					  prism.segment(*faceIndices);
					  extract.setIndices(faceIndices);
					  extract.filter(*faces);					  
					  multiViewer->updatePointCloud(faces, "Segmented Cloud");

					  //filter segmented faces
					  passThrough(*faces, passThruCloud);
					  multiViewer->updatePointCloud(passThruCloud, "Filtered Cloud");

					  meanKFilter(*faces, filteredCloud);
					  multiViewer->updatePointCloud(filteredCloud, "Convex Hull");
					}	    
					multiViewer->spinOnce(10);
				}
				multiViewer->close();
			}
			else
				ROS_INFO("Chosen hull is not planar");
		}
	}

	void cameraRollHistogram()
	{	
		PointCloudTPtr object;
		{				
			std::lock_guard<std::mutex> lock(mutex);
			//it is important to pass this->cloud by ref in order to see updates on screen
			object = PointCloudTPtr (new PointCloudT(this->cloud));
			updateCloud = false;
		}
		normalEstimation.setInputCloud(object);
		normalEstimation.setRadiusSearch(0.03);
		pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);
		normalEstimation.setSearchMethod(kdtree);
		normalEstimation.compute(*normals);

		// CRH estimation object.
		pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, CRH90> crh;
		crh.setInputCloud(object);
		crh.setInputNormals(normals);
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*object, centroid);
		crh.setCentroid(centroid);

		// Compute the CRH.
		crh.compute(*histogram);
	}
};

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "ensensor_segmentation_node"); 
	ROS_INFO("Started node %s", ros::this_node::getName().c_str());

	bool running = true;
	ros::NodeHandle nh;

	Segmentation seg(running, nh);
	seg.spawn();

	ros::shutdown();
}