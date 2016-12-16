/*
 *
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Victor Lamoine (victor.lamoine@gmail.com)
 *
 *
 * ported from pcl 1.8.
 * commit 2ca296bdd0518282fb6254ea860735fedd5306e1
 *
 */

#include <pcl/pcl_config.h>
#include <ensenso/grabber.h>
#include <ensenso/ensenso_grabber.h>

#include <pcl/exceptions.h>
#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nxLib.h> // Ensenso SDK
#include <ros/ros.h>
#include <ros/time.h>
#include <iostream>
#include <string>

using namespace ensenso;
using namespace pcl;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Handle Ensenso SDK exceptions
// This function is called whenever an exception is raised to provide details about the error
void
ensensoExceptionHandling (const NxLibException &ex,
                          std::string func_nam)
{
  PCL_ERROR ("%s: NxLib error %s (%d) occurred while accessing item %s.\n", func_nam.c_str (), ex.getErrorText ().c_str (), ex.getErrorCode (),
             ex.getItemPath ().c_str ());
  if (ex.getErrorCode () == NxLibExecutionFailed)
    {
      NxLibCommand cmd ("");
      PCL_WARN ("\n%s\n", cmd.result ().asJson (true, 4, false).c_str ());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoGrabber::EnsensoGrabber () :
  device_open_ (false),
  running_ (false)
{
  point_cloud_signal_  = createSignal<sig_cb_ensenso_point_cloud> ();
  point_cloud_signal2_ = createSignal<sig_cb_ensenso_point_cloud2> ();

  pCamera_ = new NxLibItem;

  pthread_mutex_init(&fps_mutex_,0);

  ::PCL_INFO ("Initialising nxLib\n");

  try
    {
      nxLibInitialize ();
      pRoot_ = new NxLibItem;
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "EnsensoGrabber");
      PCL_THROW_EXCEPTION (pcl::IOException, "Could not initialise NxLib.");  // If constructor fails; throw exception
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoGrabber::~EnsensoGrabber () throw ()
{
  try
    {
      stop ();
      disconnect_all_slots<sig_cb_ensenso_point_cloud> ();
      disconnect_all_slots<sig_cb_ensenso_point_cloud2> ();
      nxLibFinalize ();
      if(pRoot_) delete pRoot_;
      if(pCamera_) delete pCamera_;
      pthread_mutex_destroy(&fps_mutex_);
    }
  catch (...)
    {
      // destructor never throws
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
EnsensoGrabber::enumDevices ()
{
  int camera_count = 0;

  try
    {
      NxLibItem cams = NxLibItem ("/Cameras/BySerialNo");
      camera_count = cams.count ();

      // Print information for all cameras in the tree
      PCL_INFO ("Number of connected cameras: %d\n", camera_count);
      PCL_INFO ("Serial No    Model   Status\n");

      for (int n = 0; n < cams.count (); ++n)
        {
          PCL_INFO ("%s   %s   %s\n", cams[n][itmSerialNumber].asString ().c_str (),
                    cams[n][itmModelName].asString ().c_str (),
                    cams[n][itmStatus].asString ().c_str ());
        }
      PCL_INFO ("\n");
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "enumDevices");
    }

  return (camera_count);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::openDevice (const int device)
{
  try
    {
      if (device_open_)
        PCL_THROW_EXCEPTION (pcl::IOException, "Cannot open multiple devices!");
      
      ROS_INFO ("Opening Ensenso stereo camera id = %d\n", device);
      
      
      // Create a pointer referencing the camera's tree item, for easier access:
      (*pCamera_) = (*pRoot_)[itmCameras][itmBySerialNo][device];

      if (!pCamera_->exists () || (*pCamera_)[itmType] != valStereo)
        {
          PCL_THROW_EXCEPTION (pcl::IOException, "Please connect a single stereo camera to your computer!");
        }

      NxLibCommand open (cmdOpen);
      open.parameters ()[itmCameras] = (*pCamera_)[itmSerialNumber].asString ();
      open.execute ();
    }
  catch(pcl::IOException & ex)
    {
      ROS_ERROR("EnsensoGrabber::openDevice. could not open device: %s",ex.what());
      return false;
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "openDevice");
      return (false);
    }

  device_open_ = true;
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::closeDevice ()
{
  if (!device_open_)
    return (false);

  stop ();
  PCL_INFO ("Closing Ensenso stereo camera\n");

  try
    {
      NxLibCommand (cmdClose).execute ();
      device_open_ = false;
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "closeDevice");
      return (false);
    }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::start ()
{
  if (isRunning ())
    return;

  if (!device_open_)
    openDevice (0);


  frequency_.reset ();
  running_ = true;

  //grabber_thread_ = boost::thread (boost::bind(&ensenso::EnsensoGrabber::work, boost::ref(this)));

  int ret = pthread_create(&grabber_thread_, 0, &EnsensoGrabber::work,(void*)this);

  if(ret)
    {
      std::cerr << "EnsensoGrabber::start. could not create thread." << std::endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::stop ()
{
  if (running_)
    {
      running_ = false;  // Stop processGrabbing () callback

      //grabber_thread_.join ();  // join () waits for the thread to finish it's last iteration
      // See: http://www.boost.org/doc/libs/1_54_0/doc/html/thread/thread_management.html#thread.thread_management.thread.join
      pthread_join(grabber_thread_,0);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::isRunning () const
{
  return (running_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
EnsensoGrabber::getName () const
{
  return ("EnsensoGrabber");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::configureCapture (const bool auto_exposure,
                                  const bool auto_gain,
                                  const int bining,
                                  const float exposure,
                                  const bool front_light,
                                  const int gain,
                                  const bool gain_boost,
                                  const bool hardware_gamma,
                                  const bool hdr,
                                  const int pixel_clock,
                                  const bool projector,
                                  const int target_brightness,
                                  const std::string trigger_mode,
                                  const bool use_disparity_map_area_of_interest) const
{
  if (!device_open_)
    return (false);

  try
    {
      NxLibItem captureParams = (*pCamera_)[itmParameters][itmCapture];
      captureParams[itmAutoExposure].set (auto_exposure);
      captureParams[itmAutoGain].set (auto_gain);
      captureParams[itmBinning].set (bining);
      captureParams[itmExposure].set (exposure);
      captureParams[itmFrontLight].set (front_light);
      captureParams[itmGain].set (gain);
      captureParams[itmGainBoost].set (gain_boost);
      captureParams[itmHardwareGamma].set (hardware_gamma);
      captureParams[itmHdr].set (hdr);
      captureParams[itmPixelClock].set (pixel_clock);
      captureParams[itmProjector].set (projector);
      captureParams[itmTargetBrightness].set (target_brightness);
      captureParams[itmTriggerMode].set (trigger_mode);
      captureParams[itmUseDisparityMapAreaOfInterest].set (use_disparity_map_area_of_interest);
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "configureCapture");
      return (false);
    }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::setExtrinsicCalibration (const std::string target,
                                         const float euler_angle,
                                         const Eigen::Vector3f rotation_axis,
                                         const Eigen::Vector3f translation)
{
  if (!device_open_)
    return (false);

  try
    {
      NxLibItem calibParams = (*pCamera_)[itmLink];
      calibParams[itmTarget].set (target);
      calibParams[itmRotation][itmAngle].set (euler_angle);
      calibParams[itmRotation][itmAxis][0].set (rotation_axis[0]);
      calibParams[itmRotation][itmAxis][1].set (rotation_axis[1]);
      calibParams[itmRotation][itmAxis][2].set (rotation_axis[2]);
      calibParams[itmTranslation][0].set (translation[0]);
      calibParams[itmTranslation][1].set (translation[1]);
      calibParams[itmTranslation][2].set (translation[2]);
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "setExtrinsicCalibration");
      return (false);
    }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::getFramesPerSecondLocked (float & fps) const
{

  pthread_mutex_lock(&fps_mutex_);
  fps = frequency_.getFrequency ();
  pthread_mutex_unlock(&fps_mutex_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::openTcpPort (const int port) const
{
  try
    {
      nxLibOpenTcpPort (port);
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "openTcpPort");
      return (false);
    }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::closeTcpPort () const
{
  try
    {
      nxLibCloseTcpPort ();
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "closeTcpPort");
      return (false);
    }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
EnsensoGrabber::getTreeAsJson (const bool pretty_format) const
{
  try
    {
      return (pRoot_->asJson (pretty_format));
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "getTreeAsJson");
      return ("");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
EnsensoGrabber::getResultAsJson (const bool pretty_format) const
{
  try
    {
      NxLibCommand cmd ("");
      return (cmd.result ().asJson (pretty_format));
    }

  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "getResultAsJson");
      return ("");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoGrabber::transformationJsonToEulerAngles (const std::string &json,
                                                 double &x,
                                                 double &y,
                                                 double &z,
                                                 double &w,
                                                 double &p,
                                                 double &r)
{
  try
    {
      NxLibCommand convert (cmdConvertTransformation);
      convert.parameters ()[itmTransformation].setJson (json, false);
      convert.parameters ()[itmSplitRotation].set (valXYZ);

      convert.execute ();

      NxLibItem tf = convert.result ()[itmTransformations];
      x = tf[0][itmTranslation][0].asDouble ();
      y = tf[0][itmTranslation][1].asDouble ();
      z = tf[0][itmTranslation][2].asDouble ();
      r = tf[0][itmRotation][itmAngle].asDouble ();  // Roll
      p = tf[1][itmRotation][itmAngle].asDouble ();  // Pitch
      w = tf[2][itmRotation][itmAngle].asDouble ();  // yaW
      return (true);
    }

  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "transformationJsonToEulerAngles");
      return (false);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
EnsensoGrabber::angleAxisToTransformationJson (const double x,
                                               const double y,
                                               const double z,
                                               const double rx,
                                               const double ry,
                                               const double rz,
                                               const double alpha,
                                               const bool pretty_format)
{
  try
    {
      NxLibItem tf ("/tmpTF");
      tf[itmTranslation][0].set (x);
      tf[itmTranslation][1].set (y);
      tf[itmTranslation][2].set (z);

      tf[itmRotation][itmAngle].set (alpha);  // Angle of rotation
      tf[itmRotation][itmAxis][0].set (rx);  // X component of Euler vector
      tf[itmRotation][itmAxis][1].set (ry);  // Y component of Euler vector
      tf[itmRotation][itmAxis][2].set (rz);  // Z component of Euler vector

      std::string json = tf.asJson (pretty_format);
      tf.erase ();
      return (json);
    }

  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "angleAxisToTransformationJson");
      return ("");
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
EnsensoGrabber::eulerAnglesToTransformationJson (const double x,
                                                 const double y,
                                                 const double z,
                                                 const double w,
                                                 const double p,
                                                 const double r,
                                                 const bool pretty_format)
{
  try
    {
      NxLibCommand chain (cmdChainTransformations);
      NxLibItem tf = chain.parameters ()[itmTransformations];

      tf[0].setJson (angleAxisToTransformationJson (x, y, z, 0, 0, 1, r), false);  // Roll
      tf[1].setJson (angleAxisToTransformationJson (0, 0, 0, 0, 1, 0, p), false);  // Pitch
      tf[2].setJson (angleAxisToTransformationJson (0, 0, 0, 1, 0, 0, w), false);  // yaW

      chain.execute ();
      return (chain.result ()[itmTransformation].asJson (pretty_format));
    }
  catch (NxLibException &ex)
    {
      ensensoExceptionHandling (ex, "eulerAnglesToTransformationJson");
      return ("");
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::processGrabbing ()
{
  bool continue_grabbing = running_;


  while (continue_grabbing)
    {
      try
        {
          // Publish cloud
          int ns2 = num_slots<sig_cb_ensenso_point_cloud2> ();
          int ns1 = num_slots<sig_cb_ensenso_point_cloud> ();

          //if at least one is callback is registered, lets grab data
          if (ns1+ns2 > 0)
            {

              pthread_mutex_lock(&fps_mutex_);
              frequency_.event ();
              pthread_mutex_unlock(&fps_mutex_);


              //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA> ());
              pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());

              NxLibCommand (cmdCapture).execute ();
              ros::Time capture_time = ros::Time::now();

              // Stereo matching task
              NxLibCommand (cmdComputeDisparityMap).execute ();

              // Convert disparity map into XYZ data for each pixel
              NxLibCommand (cmdComputePointMap).execute ();

              // Get info about the computed point map and copy it into a std::vector
              std::vector<float> pointMap;
              int width, height;
              (*pCamera_)[itmImages][itmPointMap].getBinaryDataInfo (&width, &height, 0, 0, 0, 0);
              (*pCamera_)[itmImages][itmPointMap].getBinaryData (pointMap, 0);

              // Copy point cloud and convert in meters
              cloud->points.resize (height * width);
              cloud->width = width;
              cloud->height = height;
              cloud->is_dense = false;

              // Copy data in point cloud (and convert milimeters in meters)
              for (size_t i = 0; i < pointMap.size (); i += 3)
                {
                  cloud->points[i / 3].x = pointMap[i] / 1000.0;
                  cloud->points[i / 3].y = pointMap[i + 1] / 1000.0;
                  cloud->points[i / 3].z = pointMap[i + 2] / 1000.0;
                }

              if(ns1>0) point_cloud_signal_->operator () (cloud);

              if(ns2>0)
                {
                  boost::shared_ptr< ::sensor_msgs::PointCloud2 > pointCloud2 (new ::sensor_msgs::PointCloud2 );
                  pcl::toROSMsg<pcl::PointXYZ>(*cloud, *pointCloud2);
                  pointCloud2->header.stamp = capture_time;
                  pointCloud2->header.frame_id = "sensor_frame";//FIXME. look at sdk doc for correct name
                  point_cloud_signal2_->operator () (pointCloud2);
                }

            }

          continue_grabbing = running_;
        }
      catch (NxLibException &ex)
        {
          std::cout << "got an exception in processGrabbing" << std::endl;
          ensensoExceptionHandling (ex, "processGrabbing");
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::captureAll (void)
{
	std::cout << "Start stereo processing" << std::endl;
	NxLibCommand (cmdCapture).execute ();
//	NxLibCommand capture(cmdCapture);
//	capture.parameters() [itmCameras] = (*pCamera_)[itmSerialNumber].asString ();
//	capture.execute ();
//	ros::Time capture_time = ros::Time::now();

	// Stereo matching task
	NxLibCommand (cmdComputeDisparityMap).execute ();
//	NxLibCommand computeDisparityMap(cmdComputeDisparityMap);
//	computeDisparityMap.parameters() [itmCameras] = (*pCamera_)[itmSerialNumber].asString ();
//	computeDisparityMap.execute ();

	// Convert disparity map into XYZ data for each pixel
	NxLibCommand (cmdComputePointMap).execute ();
//	NxLibCommand computePointMap(cmdComputePointMap);
//	computePointMap.parameters() [itmCameras] = (*pCamera_)[itmSerialNumber].asString ();
//	computePointMap.execute ();
	std::cout << "End stereo processing" << std::endl;


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
EnsensoGrabber::getPclData (pcl::PointCloud<pcl::PointXYZ>::Ptr pcl)
{
	std::string modelNum;

    // TODO: Find a better way to determine the device id
    modelNum = (*pCamera_)[itmSerialNumber].asString();
    int device_id;
    if (modelNum.compare("140236")) {
  	  device_id = 1;
    } else {
  	  device_id = 0;
    }

	// Get info about the computed point map and copy it into a std::vector
	std::vector<float> pointMap;
	int width, height;
	(*pCamera_)[itmImages][itmPointMap].getBinaryDataInfo (&width, &height, 0, 0, 0, 0);
	(*pCamera_)[itmImages][itmPointMap].getBinaryData (pointMap, 0);

	// Copy point cloud and convert in meters
	pcl->points.resize (height * width);
	pcl->width = width;
	pcl->height = height;
	pcl->is_dense = false;

	// Copy data in point cloud (and convert millimeters in meters)
	for (size_t i = 0; i < pointMap.size (); i += 3) {
		pcl->points[i / 3].x = pointMap[i] / 1000.0;
		pcl->points[i / 3].y = pointMap[i + 1] / 1000.0;
		pcl->points[i / 3].z = pointMap[i + 2] / 1000.0;
	}

	//pcl->header.stamp = capture_time.toNSec();
}
