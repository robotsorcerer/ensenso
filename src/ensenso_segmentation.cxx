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
 	std::thread cloudDispThread,normalsDispThread;
 	unsigned long const hardware_concurrency;
 	ros::AsyncSpinner spinner;
 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	NormalEstimation normalEstimation;
	PointCloudNPtr normals;


public:
	Segmentation(bool running_)
	: updateCloud(false), save(false), running(running_), cloudName("Segmentation Cloud"),
	hardware_concurrency(std::thread::hardware_concurrency()),
	spinner(hardware_concurrency/2)
	{		
		boost::shared_ptr<visualizer> viz (new visualizer);
		viewer = viz->createViewer();

		OUT("Hardware Concurrency: " << hardware_concurrency <<
			"\t. Spinning with " << hardware_concurrency/2 << " threads");

	    cloud_sub_ = nh_.subscribe("/ensenso/cloud", 1, 
	          &Segmentation::cloudCallback, this);

	    normals = PointCloudNPtr(new PointCloudN());
	}

	~Segmentation()
	{
		viewer->close();
	}

	void spawn()
	{
		begin();
		end();
	}


private:
	void begin()
	{
		if(spinner.canStart())
		{
			spinner.start();
		}

		while(!updateCloud)
		{
		  std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		cloudDispThread = std::thread(&Segmentation::cloudDisp, this);
	}

	void end()
	{
	  spinner.stop(); 
	}

	void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& ensensoCloud)
	{
		PointCloudT cloud;
		PointCloudNPtr normals_(new PointCloudN());		

		getCloud(ensensoCloud, cloud);
		getSurfaceNormals(cloud, normals_);

		std::lock_guard<std::mutex> lock(mutex);
		this->cloud = cloud;
		this->normals = normals_;
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
	  
	  PointCloudNPtr normals( new PointCloudN);
	  // normals = this->normals;
	  pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 255, 255);	  
	  for(; running && ros::ok() ;)
	  {
	    viewer->addPointCloud(cloud_ptr, color_handler, cloudName);
	    // viewer->addPointCloudNormals<PointT, PointN>(cloud_ptr, normals, 20, 0.03, "normals");
	    /*populate the cloud viewer and prepare for publishing*/
	    if(updateCloud)
	    {   
	      std::lock_guard<std::mutex> lock(mutex); 
	      updateCloud = false;
	      // viewer->removePointCloud(cloudName);
	      viewer->updatePointCloud(cloud_ptr, cloudName);
	      // viewer->addPointCloudNormals<PointT, PointN>(cloud_ptr, normals, 20, 0.03, "normals");
	    }	    
	    viewer->spinOnce(10);
	  }
	  // viewer->close();
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
	Segmentation seg(running);

	seg.spawn();

	ros::shutdown();
}