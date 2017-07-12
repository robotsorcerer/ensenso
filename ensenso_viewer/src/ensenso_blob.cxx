#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"

//#include "getBlobsContour.cpp"

#include <ros/ros.h>
#include <ros/spinner.h>
#include <cv_bridge/cv_bridge.h>
#include <ensenso/visualizer.h>
#include <ensenso/ensenso_headers.h>

#include <ensenso/pcl_headers.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/range_image/range_image.h>  //convert black and white to range image
#include <pcl/visualization/range_image_visualizer.h>

/*Globlal namespaces and aliases*/
using namespace pathfinder;

/*aliases*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using pcl_viz = pcl::visualization::PCLVisualizer;

struct Center
  {
      cv::Point2d location;
      double radius;
      double confidence;
  };

class Receiver
{
private:
  /*aliases*/
  using imageMsgSub = message_filters::Subscriber<sensor_msgs::Image>;
  using cloudMsgSub = message_filters::Subscriber<sensor_msgs::PointCloud2>;
  using syncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,sensor_msgs::Image>;

  bool running, updateCloud, updateImage, save;
  size_t counter;
  std::ostringstream oss;

  const std::string cloudName;
  pcl::PCDWriter writer;
  std::vector<int> params;

  ros::NodeHandle nh;
  std::mutex mutex;
  cv::Mat ir;
  std::string windowName;
  const std::string basetopic;
  std::string subNameDepth;
  PointCloudT cloud;

  unsigned long const hardware_threads;
  ros::AsyncSpinner spinner;
  std::string subNameCloud, subNameIr;
  imageMsgSub subImageIr;
  cloudMsgSub subCloud;

  std::vector<std::thread> threads;

  boost::shared_ptr<pcl_viz> viewer;
  message_filters::Synchronizer<syncPolicy> sync;

  boost::shared_ptr<visualizer> viz;

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
  cloudName("ensenso_cloud"), windowName("Ensenso images"), basetopic("/ensenso"),
  hardware_threads(std::thread::hardware_concurrency()),  spinner(hardware_threads/2),
  subNameCloud(basetopic + "/cloud"), subNameIr(basetopic + "/image_combo"),
  subImageIr(nh, subNameIr, 1), subCloud(nh, subNameCloud, 1),
  sync(syncPolicy(10), subCloud, subImageIr)
  {
    sync.registerCallback(boost::bind(&Receiver::callback, this, _1, _2));
    ROS_INFO_STREAM("#Hardware Concurrency: " << hardware_threads <<
      "\t. Spinning with " << hardware_threads/4 << " threads");
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(80);
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

  void callback(const sensor_msgs::PointCloud2ConstPtr& ensensoCloud, const sensor_msgs::ImageConstPtr& ensensoImage)
  {
    cv::Mat ir;
    PointCloudT cloud;
    getImage(ensensoImage, ir);
    getCloud(ensensoCloud, cloud);

    std::lock_guard<std::mutex> lock(mutex);
    this->ir = ir;
    this->cloud = cloud;
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

  void saveCloudAndImage(const PointCloudT& clloud, const cv::Mat& image)
  {
    auto paths = pathfinder::getCurrentPath();
    boost::filesystem::path pwd = std::get<0>(paths);
    const std::string& train_imgs = std::get<3>(paths);
    const std::string& train_clouds = std::get<4>(paths);
    const std::string& test_imgs = std::get<5>(paths);
    const std::string& test_clouds = std::get<6>(paths);

    oss.str("");
    oss << counter;
    const std::string baseName = oss.str();
    const std::string cloud_id = "contour_" + baseName + "_cloud.pcd";
    const std::string imageName = "contour_" + baseName + "_image.jpg";

    ROS_INFO_STREAM("saving cloud: " << cloud_id);
    writer.writeBinary(cloud_id, clloud);
    ROS_INFO_STREAM("saving image: " << imageName);
    cv::imwrite(imageName, image, params);

    ROS_INFO_STREAM("saving complete!");
    ++counter;
  }

  void imageDisp()
  {
    cv::Mat ir;
    cv::Mat dst;
    PointCloudT cloud;
    PointCloudT newCloud;
    PointCloudT blob;
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1280, 1080) ;
    cv::SimpleBlobDetector::Params params;
    params.minArea = 5000;
    params.maxArea = INT_MAX;
    cv::SimpleBlobDetector detector(params);
    cv::namedWindow("blur", cv::WINDOW_NORMAL);
    cv::resizeWindow("blur", 1280, 1080) ;
    ros::Publisher pclPub = nh.advertise<sensor_msgs::PointCloud2>("/ensenso_viewer/contour_cloud", 5);

    for(; running && ros::ok();)
    {
      if(updateImage)
      {
        std::lock_guard<std::mutex> lock(mutex);
        ir = this->ir;
        cloud = this->cloud;
        updateImage = false;

        cv::medianBlur(ir, dst, 17);

        std::vector<cv::KeyPoint> kp;
        detector.detect(dst, kp);
        drawKeypoints(dst, kp, ir, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


    cv::Mat grayscaleImage;
    if (dst.channels() == 3)
        cvtColor(dst, grayscaleImage, CV_BGR2GRAY);
    else
      grayscaleImage = dst.clone();

    cv::Mat bImage;
    cv::threshold(grayscaleImage, bImage, 128, 255, CV_THRESH_BINARY);

    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = bImage.clone();
    cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );

    std::vector<std::vector<cv::Point> > newContours;
    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {

        Center center;
        center.confidence = 1;
        cv::Moments moms = moments(cv::Mat(contours[contourIdx]));
        if (true)
        {
            double area = moms.m00;
            if (area < 5000 || area >= std::numeric_limits<float>::max())
                continue;
        }

        if (false)
        {
            double area = moms.m00;
            double perimeter = cv::arcLength(cv::Mat(contours[contourIdx]), true);
            double ratio = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < 0.8f || ratio >= std::numeric_limits<float>::max())
                continue;
        }

        if (true)
        {
            double denominator = sqrt(pow(2 * moms.mu11, 2) + pow(moms.mu20 - moms.mu02, 2));
            const double eps = 1e-2;
            double ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < 0.1f || ratio >= std::numeric_limits<float>::max())
                continue;

            center.confidence = ratio * ratio;
        }

        if (true)
        {
            std::vector < cv::Point > hull;
            cv::convexHull(cv::Mat(contours[contourIdx]), hull);
            double area = cv::contourArea(cv::Mat(contours[contourIdx]));
            double hullArea = cv::contourArea(cv::Mat(hull));
            double ratio = area / hullArea;
            if (ratio < 0.95f || ratio >= std::numeric_limits<float>::max())
                continue;
        }

        center.location = cv::Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (true)
        {
            if (bImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != 0)
                continue;
        }

        //compute blob radius
        {
            std::vector<double> dists;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                cv::Point2d pt = contours[contourIdx][pointIdx];
                dists.push_back(norm(center.location - pt));
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }

        newContours.push_back(contours[contourIdx]);

}



   cloud = this->cloud;

    //Draw the contours
    cv::Mat contourImage(dst.size(), CV_8UC3, cv::Scalar(0,0,0));
    int j = 0;
    for (size_t idx = 0; idx < newContours.size(); idx++) {
        cv::drawContours(contourImage, newContours, idx, cv::Scalar(0, 0, 255), CV_FILLED);
    }

    PointT newpoint;

    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> data = cloud.points;
    int row;
    int col;


    for (size_t i = 0; i < data.size(); ++i){
      row++;
      if(i % 1280 == 0){
        row = 0;
        col++;
      }
    //  std::cout << i << std::endl;
      if(row < contourImage.rows && col < contourImage.cols && newContours.size() > 0){
      cv::Vec3b color = contourImage.at<cv::Vec3b>(cv::Point(row, col));
      //ROS_INFO("PBS");

      if(color[2] == 255){

        newCloud.points.resize(j + 1);
        newCloud.points[j] = data[i];
        j++;
        std::cout << newCloud.points.size() << std::endl;
      }
    }

    }



    //   detector.detect(dst, keypoints);
        //std::vector<cv::Point> contours = detector.getContours();
  /*      for (auto & kp : keypoints) {
          std::vector<int> index;
          index[0] = kp.pt.x - kp.size;
          index[1] = kp.pt.x + kp.size;
          index[2] = kp.pt.y - kp.size;
          index[3] = kp.pt.y + kp.size;
          blob = PointCloudT(cloud, index);
        }

*/

sensor_msgs::PointCloud2 pcl2_msg;
pcl::toROSMsg(newCloud, pcl2_msg);
pcl2_msg.header.stamp = ros::Time::now();
pcl2_msg.header.frame_id = "contour_cloud";
pclPub.publish(pcl2_msg);

        cv::imshow(windowName, ir);
        cv::imshow("blur", contourImage);

        int key = cv::waitKey(1);
        switch(key & 0xFF)
        {
          case 27:
          case 'q':
            running = false;
            break;
          case ' ':
          case 's':
            saveCloudAndImage(newCloud, ir);
            save = true;
            break;
        }
      }
      }


/*
      int key = cv::waitKey(1);
      switch(key & 0xFF)
      {
        case 27:
        case 'q':
          running = false;
          break;
        case ' ':
        case 's':
          saveCloudAndImage(newCloud, ir);
          save = true;
          break;
      }
    }*/
    cv::destroyAllWindows();
    cv::waitKey(100);
  }

  void
  setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
  {
    Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
    viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                              look_at_vector[0], look_at_vector[1], look_at_vector[2],
                              up_vector[0], up_vector[1], up_vector[2]);
  }

  void cloudDisp()
  {
    viz = boost::shared_ptr<visualizer> (new visualizer());
    viewer = boost::shared_ptr<pcl_viz> (new pcl_viz);
    viewer= viz->createViewer();

    // PointCloudT cloud  = this->cloud;
    PointCloudT::Ptr cloud_ptr (&this->cloud);

    // Visualize the image.
    // range_image_ptr = boost::shared_ptr<pcl::RangeImage> (new pcl::RangeImage);
    // pcl::RangeImage& range_image = *range_image_ptr;
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_color_handler (range_image_ptr, 225, 155, 155);

    //do range image
    Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(cloud_ptr->sensor_origin_[0],
								 cloud_ptr->sensor_origin_[1],
								 cloud_ptr->sensor_origin_[2])) *
								 Eigen::Affine3f(cloud_ptr->sensor_orientation_);

    // Noise level. If greater than 0, values of neighboring points will be averaged.
   	// This would set the search radius (e.g., 0.03 == 3cm).
   	float noiseLevel = 0.0f;
   	// Minimum range. If set, any point closer to the sensor than this will be ignored.
   	float minimumRange = 0.0f;
   	// Border size. If greater than 0, a border of "unobserved" points will be left
   	// in the image when it is cropped.
   	int borderSize = 1;

    // range_image.createFromPointCloud(*cloud_ptr, angularResolutionX, angularResolutionY,
		// 							maxAngleX, maxAngleY, sensorPose, pcl::RangeImage::CAMERA_FRAME,F
		// 							noiseLevel, minimumRange, borderSize);


    pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler (cloud_ptr, 255, 150, 155);
    cv::Mat ir = this->ir;
    viewer->addPointCloud(cloud_ptr, color_handler, cloudName);
    // setViewerPose(*viewer, range_image.getTransformationToWorldSystem());

    for(; running && ros::ok() ;)
    {
      /*populate the cloud viewer and prepare for publishing*/
      if(updateCloud)
      {
        std::lock_guard<std::mutex> lock(mutex);
        updateCloud = false;

        // ROS_INFO_STREAM(" width: " << cloud_ptr->width << " | height: " << cloud_ptr->height);

        // sensorPose = viewer->getViewerPose();
        // range_image.createFromPointCloud(*cloud_ptr, angularResolutionX, angularResolutionY,
    		// 							maxAngleX, maxAngleY, sensorPose, pcl::RangeImage::CAMERA_FRAME,
    		// 							noiseLevel, minimumRange, borderSize);
        viewer->updatePointCloud(cloud_ptr, color_handler, cloudName);
      }
      if(save)
      {
        save = false;
      //  saveCloudAndImage(*cloud_ptr, ir);
      }
      viewer->spinOnce(10);
    }
    viewer->close();
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ensensor_viewer_node");

  ROS_INFO_STREAM("Started node " << ros::this_node::getName().c_str());

  Receiver r;
  r.run();

  if(!ros::ok())
  {
    return 0;
  }

  ros::shutdown();
}
