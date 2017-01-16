#include <ros/spinner.h>

#include <ensenso/visualizer.h>
#include <ensenso/pcl_headers.h>
#include <sensor_msgs/PointCloud2.h>

#include <mutex>
#include <thread>

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
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	NormalEstimation normalEstimation;
	PointCloudNPtr normals;

	boost::shared_ptr<visualizer> viz;
	std::vector<std::thread> threads;
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
	    threads.push_back(std::thread(&Segmentation::cloudDisp, this));
	    std::for_each(threads.begin(), threads.end(), \
	                  std::mem_fn(&std::thread::join)); 
	    cloudDispThread = std::thread(&Segmentation::cloudDisp, this);
	    ROS_INFO("pushed threads");
	}

	void end()
	{
	  spinner.stop(); 
	  running = false;
	}

	void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& ensensoCloud)
	{
		PointCloudT cloud;
		PointCloudNPtr normals_(new PointCloudN());		

		getCloud(ensensoCloud, cloud);
		// getSurfaceNormals(cloud, normals_);

		std::lock_guard<std::mutex> lock(mutex);
		this->cloud = cloud;
		// this->normals = normals_;
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

	void getSurfaceNormals(const PointCloudT& cloud, PointCloudNPtr normals)
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