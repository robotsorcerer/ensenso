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

/*pcl and cv headers*/
#include <ensenso/ensenso_headers.h>

/*typedefs*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using  CloudViewer = pcl::visualization::CloudViewer;
using PairOfImages =  std::pair<pcl::PCLImage, pcl::PCLImage>;  //for the Ensenso grabber callback 

#define OUT(__x__) std::cout << __x__ << std::endl;
/*pointers*/
std::shared_ptr<CloudViewer> viewer_ptr;
pcl::EnsensoGrabber::Ptr ensenso_ptr;

void grabberCallback (const boost::shared_ptr<PointCloudT>& cloud, const boost::shared_ptr<PairOfImages>& images)
{
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
}

int main (void)
{
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

  return EXIT_SUCCESS;
}

