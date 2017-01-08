/*  
*   MIT License
*   
*   Copyright (c) December 2016 
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*   
*   The above copyright notice and this permission notice shall be included in all
*   copies or substantial portions of the Software.
*   
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*   SOFTWARE.
*
* 
*  Author: Olalekan P. Ogunmolu
*/
#include <ros/ros.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <algorithm>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl_conversions/pcl_conversions.h>

/*pcl and cv headers*/
#include <ensenso/ensenso_headers.h>

/*typedefs*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using  CloudViewer = pcl::visualization::CloudViewer;
using PairOfImages =  std::pair<pcl::PCLImage, pcl::PCLImage>;  //for the Ensenso grabber callback 

/*message filters typedefs*/
/*using imageMessageSub = message_filters::Subscriber<sensor_msgs::Image> ;
using camInfoSub = message_filters::Subscriber<sensor_msgs::CameraInfo> ;
using syncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ;
// using syncPolicy =  message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ;

imageMessageSub subImageColor, subImageDepth;
camInfoSub subInfoCam, subInfoDepth;
message_filters::Synchronizer<syncPolicy> sync;*/


#define OUT(__x__) std::cout << __x__ << std::endl;
/*pointers*/
std::shared_ptr<CloudViewer> viewer_ptr;
pcl::EnsensoGrabber::Ptr ensenso_ptr;

void grabberCallback (const boost::shared_ptr<PointCloudT>& cloud, const boost::shared_ptr<PairOfImages>& images)
{
  ros::NodeHandle nh;
  ros::Publisher pclPub;
  sensor_msgs::PointCloud2 pcl2_cloud;   //msg to be displayed in rviz
  image_transport::ImageTransport it(nh);
  PointT rviz_points;           //since rviz are scaled in meters, we need a conversion
  pcl::PointCloud<PointT>::Ptr rviz_cloud_ptr(new pcl::PointCloud<PointT>);
  image_transport::Publisher imagePub = it.advertise("camera/ensenso/mono", 1);

  if (!viewer_ptr->wasStopped ())  viewer_ptr->showCloud (cloud);
  unsigned char *l_image_array = reinterpret_cast<unsigned char *> (&images->first.data[0]);
  unsigned char *r_image_array = reinterpret_cast<unsigned char *> (&images->second.data[0]);

  std::cout << "Encoding: " << images->first.encoding << std::endl;
  
  int type = getOpenCVType (images->first.encoding);
  cv::Mat l_image (images->first.height, images->first.width, type, l_image_array);
  cv::Mat r_image (images->first.height, images->first.width, type, r_image_array);
  cv::Mat im (images->first.height, images->first.width * 2, type);

  im.adjustROI (0, 0, 0, -images->first.width);
  l_image.copyTo (im);
  im.adjustROI (0, 0, -images->first.width, images->first.width);
  r_image.copyTo (im);
  im.adjustROI (0, 0, images->first.width, 0);
  cv::imshow ("Ensenso images", im);
  cv::waitKey (2);

  //prepare image and pcl to be published for rospy
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), images->first.encoding, im).toImageMsg();
  //prepare cloud for rospy 
  pcl::toROSMsg(*cloud, pcl2_cloud);
  pcl2_cloud.header.stamp = ros::Time::now();
  pcl2_cloud.header.frame_id = "rospy_cloud";

  // ros::Rate rate(5);
  if(ros::ok()){
    imagePub.publish(msg);
    pclPub.publish(pcl2_cloud);
    ros::spinOnce();
    // rate.sleep();
  }
}

int main (int argc, char** argv)
{
  ros::init(argc, argv, "ensensor_publisher_node", ros::init_options::AnonymousName);

  viewer_ptr.reset (new CloudViewer ("Ensenso viewer"));
  ensenso_ptr.reset (new pcl::EnsensoGrabber);
  ensenso_ptr->openTcpPort ();
  ensenso_ptr->openDevice ();
  ensenso_ptr->enumDevices ();

  //ensenso_ptr->initExtrinsicCalibration (5); // Disable projector if you want good looking images.

  boost::function<void(const boost::shared_ptr<PointCloudT>&, const boost::shared_ptr<PairOfImages>&)> f = boost::bind(&grabberCallback, _1, _2);
  ensenso_ptr->registerCallback (f);

  cv::namedWindow("Ensenso images", cv::WINDOW_NORMAL);
  cv::resizeWindow("Ensenso images", 640, 480) ;
  ensenso_ptr->start ();

  while (!viewer_ptr->wasStopped ())
  {

    ROS_INFO("FPS: %f\n", ensenso_ptr->getFramesPerSecond ());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  ensenso_ptr->stop ();
  ensenso_ptr->closeDevice ();
  ensenso_ptr->closeTcpPort ();

  ros::shutdown();

  return EXIT_SUCCESS;
}

