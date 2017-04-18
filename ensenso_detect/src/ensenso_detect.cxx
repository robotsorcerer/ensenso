#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <iostream>
#include <cmath>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <cv_bridge/cv_bridge.h>
#include <ensenso/visualizer.h>
#include <ensenso/ensenso_headers.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

/*Globlal namespaces and aliases*/
using namespace pathfinder;

/*aliases*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using pcl_viz = pcl::visualization::PCLVisualizer;

class Receiver
{
private:
  /*aliases*/
  using imageMsgSub = message_filters::Subscriber<sensor_msgs::Image>;
  using cloudMsgSub = message_filters::Subscriber<sensor_msgs::PointCloud2>;
  using syncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                  sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>;
  bool running, updateCloud, updateImage, save;
  size_t counter;
  std::ostringstream oss;

  const std::string cloudName;
  pcl::PCDWriter writer;
  std::vector<int> params;

  ros::NodeHandle nh;
  std::mutex mutex;
  cv::Mat ir, leftIr, rightIr;
  std::string windowName;
  const std::string basetopic;
  std::string subNameDepth;
  PointCloudT cloud;

  unsigned long const hardware_threads;
  ros::AsyncSpinner spinner;
  std::string subNameCloud, subNameIr, subNameIrLeft, subNameIrRight;
  imageMsgSub subImageIr, subImageIrLeft, subImageIrRight;
  cloudMsgSub subCloud;

  std::vector<std::thread> threads;

  boost::shared_ptr<pcl_viz> viewer;
  message_filters::Synchronizer<syncPolicy> sync;

  boost::shared_ptr<visualizer> viz;
  bool showLeftImage, showRightImage;

  // Parameters needed by the range image object:
  // Angular resolution is the angular distance between pixels.
  // Kinect: 57° horizontal FOV, 43° vertical FOV, 640x480 (chosen here).
  // Xtion: 58° horizontal FOV, 45° vertical FOV, 640x480.
  float angularResolutionX = (float)(57.0f / 640.0f * (M_PI / 180.0f));
  float angularResolutionY = (float)(43.0f / 480.0f * (M_PI / 180.0f));
  // Maximum horizontal and vertical angles. For example, for a full panoramic scan,
  // the first would be 360º. Choosing values that adjust to the real sensor will
  // decrease the time it takes, but don't worry. If the values are bigger than
  // the real ones, the image will be automatically cropped to discard empty zones.
  float maxAngleX = (float)(60.0f * (M_PI / 180.0f));
  float maxAngleY = (float)(50.0f * (M_PI / 180.0f));
  // Range image object.
  boost::shared_ptr<pcl::RangeImage> range_image_ptr;
  pcl::RangeImage rangeImage;


public:
  //constructor
  Receiver()
  : updateCloud(false), updateImage(false), save(false), counter(0),
  cloudName("detector_cloud"), windowName("Ensenso images"), basetopic("/ensenso"),
  hardware_threads(std::thread::hardware_concurrency()),  spinner(hardware_threads/2),
  subNameCloud(basetopic + "/cloud"), subNameIr(basetopic + "/image_combo"),
  subNameIrLeft(basetopic + "/left/image"), subNameIrRight(basetopic + "/right/image"),
  subImageIr(nh, subNameIr, 1), subImageIrLeft(nh, subNameIrLeft, 1),
  subImageIrRight(nh, subNameIrRight, 1), subCloud(nh, subNameCloud, 1),
  sync(syncPolicy(10), subCloud, subImageIr, subImageIrLeft, subImageIrRight)
  {
    sync.registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
    ROS_INFO_STREAM("#Hardware Concurrency: " << hardware_threads <<
      "\t. Spinning with " << hardware_threads/4 << " threads");
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(3);

    //set ros parameters
    nh.setParam("showLeftImage", false);
    nh.setParam("showRightImage", false);
  }
  //destructor
  ~Receiver()
  {
    viz.reset();
    viewer.reset();
  }

  Receiver(Receiver const&) =delete;
  Receiver& operator=(Receiver const&) = delete;


  void run()
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
    running = true;
    while(!updateImage || !updateCloud)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    nh.getParam("showLeftImage", showLeftImage);
    nh.getParam("showRightImage", showRightImage);

    //spawn the threads
    threads.push_back(std::thread(&Receiver::cloudDisp, this));
    threads.push_back(std::thread(&Receiver::imageDisp, this));
    //call join on each thread in turn
    std::for_each(threads.begin(), threads.end(), \
                  std::mem_fn(&std::thread::join));
  }

  void end()
  {
    spinner.stop();
    running = false;
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr& ensensoCloud, const sensor_msgs::ImageConstPtr& ensensoImage,
                const sensor_msgs::ImageConstPtr& leftImage, const sensor_msgs::ImageConstPtr& rightImage)
  {
    cv::Mat ir, leftIr, rightIr;
    PointCloudT cloud;
    getImage(ensensoImage, ir);
    getCloud(ensensoCloud, cloud);
    if(showLeftImage)
      getImage(leftImage, leftIr);
    if(showRightImage)
      getImage(rightImage, rightIr);

    std::lock_guard<std::mutex> lock(mutex);
    this->ir = ir;
    this->cloud = cloud;
    if(showLeftImage)
      this->leftIr = leftIr;
    if(showRightImage)
      this->rightIr = rightIr;
    updateImage = true;
    updateCloud = true;
  }

  void getImage(const sensor_msgs::ImageConstPtr msgImage, cv::Mat &image) const
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msgImage, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(image);
  }

  void getCloud(const sensor_msgs::PointCloud2ConstPtr cb_cloud, PointCloudT& pcl_cloud) const
  {
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*cb_cloud, pcl_pc);
    pcl::fromPCLPointCloud2(pcl_pc, pcl_cloud);
  }

  void imageDisp()
  {
    cv::Mat ir, leftIr, rightIr;
    PointCloudT cloud;
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 640, 480) ;

    const std::string leftWindowName = "leftImage";
    const std::string rightWindowName = "rightImage";

    //left image window
    if(showLeftImage)
    {
      cv::namedWindow(leftWindowName, cv::WINDOW_NORMAL);
      cv::resizeWindow(leftWindowName, 640, 480);
    }

    //right image window
    if(showRightImage)
    {
      cv::namedWindow(rightWindowName, cv::WINDOW_NORMAL);
      cv::resizeWindow(rightWindowName, 640, 480);
    }

    for(; running && ros::ok();)
    {
      if(updateImage)
      {
        std::lock_guard<std::mutex> lock(mutex);
        ir = this->ir;
        if(showLeftImage)
          leftIr = this->leftIr;
        if(showRightImage)
          rightIr = this->rightIr;
        cloud = this->cloud;
        updateImage = false;

        cv::imshow(windowName, ir);
        if(showLeftImage)
          cv::imshow(leftWindowName, leftIr);
        if(showRightImage)
          cv::imshow(rightWindowName, rightIr);
      }

      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
        case 27:
        case 'q':
          running = false;
          break;
        case ' ':
        case 's':
          // saveCloudAndImage(cloud, ir);
          save = true;
          break;
      }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void cloudDisp()
  {
    viz = boost::shared_ptr<visualizer> (new visualizer());
    viewer = boost::shared_ptr<pcl_viz> (new pcl_viz);
    viewer= viz->createViewer();

    // PointCloudT cloud  = this->cloud;
    PointCloudT::Ptr cloud_ptr (&this->cloud);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 150, 155);
    cv::Mat ir = this->ir;
    viewer->addPointCloud(cloud_ptr, color_handler, cloudName);

    for(; running && ros::ok() ;)
    {
      /*populate the cloud viewer and prepare for publishing*/
      if(updateCloud)
      {
        std::lock_guard<std::mutex> lock(mutex);
        updateCloud = false;
        viewer->updatePointCloud(cloud_ptr, color_handler, cloudName);
      }
      if(save)
      {
        save = false;
        // saveCloudAndImage(*cloud_ptr, ir);
      }
      viewer->spinOnce(10);
    }
    viewer->close();
  }
};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ensensor_face_detection_node", ros::init_options::AnonymousName);

  ROS_INFO_STREAM("Started node " << ros::this_node::getName().c_str());

  Receiver r;
  r.run();

  if(!ros::ok())
  {
    return 0;
  }
  ros::shutdown();
}
