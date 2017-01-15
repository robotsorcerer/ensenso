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
#include <memory>
#include <algorithm>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <pcl_conversions/pcl_conversions.h>

/*pcl and cv headers*/
#include <ensenso/ensenso_headers.h>

/*typedefs*/
using PairOfImages =  std::pair<pcl::PCLImage, pcl::PCLImage>;  //for the Ensenso grabber callback 
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

/*pointers*/
pcl::EnsensoGrabber::Ptr ensenso_ptr;

/*Globals*/
sensor_msgs::PointCloud2 pcl2_msg;   //msg to be displayed in rviz

void callback (const PointCloudT::Ptr& cloud, \
          const boost::shared_ptr<PairOfImages>& images)
{
  ros::NodeHandle nh;
  // image_transport::ImageTransport it(nh);
  // image_transport::Publisher imagePub = it.advertise("/camera/image", 10);
  ros::Publisher pclPub = nh.advertise<sensor_msgs::PointCloud2>("/ensenso_cloudy", 10);

  //prepare cloud for rospy publishing
  pcl::toROSMsg(*cloud, pcl2_msg);
  pcl2_msg.header.stamp = ros::Time::now();
  pcl2_msg.header.frame_id = "ensenso_cloud";

  /*Process Image and prepare for publishing*/
  unsigned char *l_image_array = reinterpret_cast<unsigned char *> (&images->first.data[0]);
  unsigned char *r_image_array = reinterpret_cast<unsigned char *> (&images->second.data[0]);

  ROS_INFO_STREAM("Encoding: " << images->first.encoding);
  ROS_INFO_STREAM("Cloud size: " << cloud->height * cloud->width);
  
  int type = getOpenCVType (images->first.encoding);
  cv::Mat l_image (images->first.height, images->first.width, type, l_image_array);
  cv::Mat r_image (images->first.height, images->first.width, type, r_image_array);
  cv::Mat im (images->first.height, images->first.width * 2, type);

  im.adjustROI (0, 0, 0, -0.5*images->first.width);
  l_image.copyTo (im);
  im.adjustROI (0, 0, -0.5*images->first.width, 0.5*images->first.width);
  r_image.copyTo (im);
  im.adjustROI (0, 0, 0.5*images->first.width, 0);
  //prepare image and pcl to be published for rospy
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), images->first.encoding, im).toImageMsg();

  /*Publish the image and cloud*/
  pclPub.publish(pcl2_msg);
  // imagePub.publish(msg);  
}

int main (int argc, char** argv)
{
  ros::init(argc, argv, "ensensor_bridge_node"); 
  ros::start();
  ROS_INFO("Started node %s", ros::this_node::getName().c_str());

  ensenso_ptr.reset (new pcl::EnsensoGrabber);
  ensenso_ptr->openTcpPort();
  ensenso_ptr->openDevice();
  ensenso_ptr->enumDevices();

  //ensenso_ptr->initExtrinsicCalibration (5); // Disable projector if you want good looking images.
  boost::function<void(const PointCloudT::Ptr&, const boost::shared_ptr<PairOfImages>&)> f \
                                = boost::bind(&callback, _1, _2);

  ensenso_ptr->start ();

  ros::Rate rate(5);
  while(ros::ok())
  {    
    ensenso_ptr->registerCallback (f);
    ros::spin();
    rate.sleep();
  }

  ensenso_ptr->stop ();
  ensenso_ptr->closeDevice ();
  ensenso_ptr->closeTcpPort ();

  return EXIT_SUCCESS;
}

