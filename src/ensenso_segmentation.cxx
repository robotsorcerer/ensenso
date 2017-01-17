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
 	unsigned long const hardware_concurrency;
 	ros::AsyncSpinner spinner;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, segViewer;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> multiViewer;
 	NormalEstimation normalEstimation;
	PointCloudNPtr normals;

	boost::shared_ptr<visualizer> viz;
	std::vector<std::thread> threads;	
	float distThreshold; //distance at which to apply the threshold
	float zmin, zmax;	//minimum and maximum distance to extrude the prism object
public:
	Segmentation(bool running_, ros::NodeHandle nh)
	: nh_(nh), updateCloud(false), save(false), running(running_), cloudName("Segmentation Cloud"),
	hardware_concurrency(std::thread::hardware_concurrency()), distThreshold(0.712), zmin(0.2f), zmax(0.6953f),
	spinner(hardware_concurrency/2)
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

	void cloudDisp() 
	{
	  const PointCloudT& cloud  = this->cloud;    
	  PointCloudT::ConstPtr cloud_ptr (&cloud);
	  PointCloudNPtr normals(new PointCloudN);
	  normals = this->normals;  
	  
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
		// Objects for storing the point clouds.
		pcl::PointCloud<pcl::PointXYZ>::Ptr segCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr faces(new pcl::PointCloud<pcl::PointXYZ>);
		//Get Plane Model
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		//indices of segmented face
		pcl::PointIndices::Ptr faceIndices(new pcl::PointIndices);
		// Prism object.
		pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;

		//viewer for segmentation and stuff
		multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));   
		multiViewer->initCameraParameters ();
		int v1(0);
		multiViewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		multiViewer->setBackgroundColor (0, 0, 0, v1);
		multiViewer->addText("Original Cloud", 10, 10, "v1 text", v1);
		int v2(0);
		multiViewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		multiViewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
		multiViewer->addText("Segmented Cloud", 10, 10, "v2 text", v2);

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

			//ensure hull is 2 -d
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

				multiViewer->addPointCloud(segCloud, "Original Cloud", v1);
				multiViewer->addPointCloud(faces, "Segmented Cloud", v2);

				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Cloud");

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
					}	    
					multiViewer->spinOnce(10);
				}
				multiViewer->close();
			}
			else
				ROS_INFO("Chosen hull is not planar");
		}
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