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
	hardware_concurrency(std::thread::hardware_concurrency()),
	spinner(hardware_concurrency/2)
	{	
	    cloud_sub_ = nh.subscribe("ensenso/cloud", 10, &Segmentation::cloudCallback, this); 
		    normals = PointCloudNPtr(new PointCloudN());
	}

	~Segmentation()
	{
		// viewer->close();
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
			ROS_INFO("started spinner");
		}

		while(!updateCloud)
		{
		  std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		//spawn the threads
	    threads.push_back(std::thread(&Segmentation::planeSeg, this));
	    threads.push_back(std::thread(&Segmentation::headSeg, this));

	    std::for_each(threads.begin(), threads.end(), \
	                  std::mem_fn(&std::thread::join)); 

	    // cloudDispThread = std::thread(&Segmentation::cloudDisp, this);
	    normals = PointCloudNPtr (new PointCloudN());	
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
		// getSurfaceNormals(cloud, this->normals);

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
	  viewer->addPointCloudNormals<PointT, PointN>(cloud_ptr, normals, 20, 0.03, "normals");

	  for(; running && ros::ok() ;)
	  {
	    /*populate the cloud viewer and prepare for publishing*/
	    if(updateCloud)
	    {   
	      std::lock_guard<std::mutex> lock(mutex); 
	      updateCloud = false;
	      viewer->updatePointCloud(cloud_ptr, cloudName);
	      // viewer->addPointCloudNormals<PointT, PointN>(cloud_ptr, normals, 20, 0.03, "normals");
	    }	    
	    viewer->spinOnce(10);
	  }
	  viewer->close();
	}

	void getSurfaceNormals(const PointCloudT& cloud, const PointCloudNPtr normals)
	{
		PointCloudT::ConstPtr cloud_ptr (&cloud);
		normalEstimation.setInputCloud(cloud_ptr);
		normalEstimation.setRadiusSearch(0.03); //use all neighbors in a radius of 3cm.
		// use a kd-tree to search for neighbors.
		pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
		normalEstimation.setSearchMethod(kdtree);

		// Calculate the normals.
		normalEstimation.compute(*normals);
	}

	void planeSeg()
	{
		// Objects for storing the point clouds.
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);

		//Get Plane Model
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients (true);
		/*Set the maximum number of iterations before giving up*/
		seg.setMaxIterations(100);
		// Mandatory
		seg.setModelType (pcl::SACMODEL_PLANE); 
		seg.setMethodType (pcl::SAC_RANSAC);
		/*set the distance to the model threshold to use*/
		distThreshold = 0.712;
		seg.setDistanceThreshold (distThreshold);	//as measured
		{
			std::lock_guard<std::mutex> lock(mutex);
			*cloud = this->cloud;			
			updateCloud = false;	
		}

		seg.setInputCloud (cloud);
		seg.segment (*inliers, *coefficients);

		if(inliers->indices.size() == 0)
			ROS_INFO("Could not find plane in scene");
		else
		{
			// Copy the points of the plane to a new cloud.
			pcl::ExtractIndices<pcl::PointXYZ> extract;
			extract.setInputCloud(cloud);
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
				// Prism object.
				pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
				prism.setInputCloud(cloud);
				prism.setInputPlanarHull(convexHull);
				// First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
				// Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the objects you expect.
				// prism.setHeightLimits(0.1f, 0.5f);
				zmin = 0.2f; zmax = 0.4953f+0.2f;
				prism.setHeightLimits(zmin, zmax);
				pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);

				prism.segment(*objectIndices);

				// Get and show all points retrieved by the hull.
				extract.setIndices(objectIndices);
				extract.filter(*objects);

				multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));   
				multiViewer->initCameraParameters ();

				int v1(0);
				multiViewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
				multiViewer->setBackgroundColor (0, 0, 0, v1);
				multiViewer->addText("Original Cloud: 0.01", 10, 10, "v1 text", v1);
				multiViewer->addPointCloud(cloud, "Original Cloud", v1);

				int v2(0);
				multiViewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
				multiViewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
				multiViewer->addText("Segmented Cloud", 10, 10, "v2 text", v2);
				multiViewer->addPointCloud(objects, "Segmented Cloud", v2);

				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Cloud");

				while (running && ros::ok())
				{
					/*populate the cloud viewer and prepare for publishing*/
					if(updateCloud)
					{   
					  std::lock_guard<std::mutex> lock(mutex); 
					  updateCloud = false;
					  multiViewer->updatePointCloud(cloud, "Original Cloud");
					  multiViewer->updatePointCloud(objects, "Segmented Cloud");
					}	    
					multiViewer->spinOnce(10);
				}
				multiViewer->close();
			}
			else
				ROS_INFO("Chosen hull is not planar");
		}
	}

	void headSeg()
	{
		// Objects for storing the point clouds.
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr head(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);

		//Get Plane Model
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		//set min and max radius for sphere discovery
		const double& min_radius = 0.01;
		const double& max_radius = 0.03;
		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients (true);
		// Mandatory
		seg.setModelType (pcl::SACMODEL_SPHERE); 
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setRadiusLimits(min_radius, max_radius);
		// seg.setDistanceThreshold (0.01);
		//set axis along which to search for a model perpendicular to
		Eigen::Vector3f ax;
		ax << 0, 0, 1;
		seg.setAxis(ax);

		{
			// mutex.lock();
			std::lock_guard<std::mutex> lock(mutex);
			*cloud = this->cloud;			
			updateCloud = false;	
		}

		seg.setInputCloud (cloud);
		//inliers: resultant pt indices that support found model
		//coefficients: resultant model coefficients
		seg.segment (*inliers, *coefficients);

		if(inliers->indices.size() == 0)
			ROS_INFO("Could not find sphere in scene");
		else
		{
			// Copy the points of the spherepcl::extract indices to a new cloud.
			pcl::ExtractIndices<pcl::PointXYZ> extract(true); //init with true will always allow to extract removed indices
			extract.setInputCloud(cloud);
			extract.setIndices(inliers);
			//the resulting cloud that contains the indices of the head spherical model
			extract.filter(*head);

			//retrieve convex hull
			pcl::ConvexHull<PointT> hull;
			hull.setInputCloud(head);

			//ensure hull is 3 -d
			hull.setDimension(3);
			/*compute a convex hull for all points given*/
			hull.reconstruct(*convexHull);
			/*compute the area volume of the head and output the result
			to console*/
			hull.setComputeAreaVolume(true);

			//redundant check
			if (hull.getDimension() == 3)
			{
				// Prism object.
				pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
				prism.setInputCloud(cloud);
				prism.setInputPlanarHull(convexHull);
				// First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
				// Second parameter: maximum Z value, set to 50cm. Tune it according to the height of the objects you expect.
				prism.setHeightLimits(0.2f, 0.6f);
				pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);

				prism.segment(*objectIndices);

				// Get and show all points retrieved by the hull.
				extract.setIndices(objectIndices);
				extract.filter(*objects);

				multiViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("Multiple Viewer"));   
				multiViewer->initCameraParameters ();

				int v1(0);
				multiViewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
				multiViewer->setBackgroundColor (0, 0, 0, v1);
				multiViewer->addText("Original Cloud: 0.01", 10, 10, "v1 text", v1);
				multiViewer->addPointCloud(cloud, "Original Cloud", v1);

				int v2(0);
				multiViewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
				multiViewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
				multiViewer->addText("Segmented Cloud", 10, 10, "v2 text", v2);
				multiViewer->addPointCloud(objects, "Segmented Cloud", v2);

				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Original Cloud");
				multiViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Segmented Cloud");

				while (running && ros::ok())
				{
					/*populate the cloud viewer and prepare for publishing*/
					if(updateCloud)
					{   
					  std::lock_guard<std::mutex> lock(mutex); 
					  updateCloud = false;
					  multiViewer->updatePointCloud(cloud, "Original Cloud");
					  multiViewer->updatePointCloud(objects, "Segmented Cloud");
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