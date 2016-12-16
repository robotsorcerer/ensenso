#include "ensenso/ensenso_interface.h"
#include <ros/ros.h>
#include <pcl/exceptions.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <boost/filesystem.hpp>

void getFileNameMatch(std::string searchString, std::string dirPath, std::string *filePathFound);

using namespace ensenso;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Handle Ensenso SDK exceptions
// This function is called whenever an exception is raised to provide details about the error
void ensensoExceptionHandling(const NxLibException &ex, std::string func_nam)
{
  ROS_ERROR("%s: NxLib error %s (%d) occurred while accessing item %s.\n", func_nam.c_str(), ex.getErrorText().c_str(),
            ex.getErrorCode(), ex.getItemPath().c_str());
  if (ex.getErrorCode() == NxLibExecutionFailed) {
    NxLibCommand cmd("");
    PCL_WARN("\n%s\n", cmd.result().asJson(true, 4, false).c_str());
  }
}

/***************************************************************************************************************/
/* CLASS EnsensoCameraInterface */
/***************************************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoCameraInterface::EnsensoCameraInterface() :
    device_open_(false), device_num_(0)
{
  pCamera_ = new NxLibItem;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoCameraInterface::~EnsensoCameraInterface()
{
  nxLibFinalize();
  if (pCamera_)
    delete pCamera_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::openDevice(void)
{
  try {
    if (device_open_) {
      PCL_THROW_EXCEPTION(pcl::IOException, "Cannot open multiple devices!");
    }

    ROS_INFO("Opening Ensenso stereo camera id = %d", device_num_);

    if (!pCamera_->exists() || (*pCamera_)[itmType] != valStereo) {
      PCL_THROW_EXCEPTION(pcl::IOException, "Please connect a single stereo camera to your computer!");
    }

    NxLibCommand open(cmdOpen);
    open.parameters()[itmCameras] = (*pCamera_)[itmSerialNumber].asString();
    open.execute();
    ROS_INFO("Ensenso stereo camera id = %d opened.", device_num_);
  }
  catch (pcl::IOException & ex) {
    ROS_ERROR("EnsensoInterface::openDevice.  Could not open device: %s", ex.what());
    return (false);
  }
  catch (NxLibException &ex) {
    ensensoExceptionHandling(ex, "openDevice");
    return (false);
  }

  device_open_ = true;
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::closeDevice(void)
{
  if (!device_open_) {
    return (false);
  }

  ROS_INFO("Closing Ensenso stereo camera %d\n", device_num_);
  try {
    NxLibCommand(cmdClose).execute();
    device_open_ = false;
  }
  catch (NxLibException &ex) {
    ensensoExceptionHandling(ex, "closeDevice");
    return (false);
  }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoCameraInterface::configureCapture (const bool auto_exposure,
                                  const bool auto_gain,
                                  const int binning,
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
      captureParams[itmBinning].set (binning);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoCameraInterface::configureCameraFromJson(std::string jsonFilePath)
{
  bool pretty_format = true;
  std::ifstream inFile;
  std::stringstream jsonTreeStr;

  inFile.open(jsonFilePath.c_str());
  if (!inFile.is_open()) {
    //ui_report_status(gyges_status::ENSENSOS_CFG_FILE_FAILED_TO_OPEN);
    ROS_ERROR("Failed to open json configuration file in EnsensoSystemInterface::configureCameraFromJson");
    return (false);
  } else {
    /* Read the entire json file contents into a string */
    jsonTreeStr << inFile.rdbuf();

    try {
      /* Put the json string contents into the camera item on the NxLib tree */
      (*pCamera_)[itmParameters].setJson(jsonTreeStr.str(), NXLIBTRUE);
    }
    catch (NxLibException &ex) {
      ensensoExceptionHandling(ex, "configureCameraFromJson");
      return (false);
    }
    ROS_INFO("Camera configured.");
    return (true);
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::triggerSensor(void)
{
  std::cout << "Start stereo processing" << std::endl;
  NxLibCommand capture(cmdCapture);
  capture.parameters()[itmCameras] = (*pCamera_)[itmSerialNumber].asString();
  capture.execute();
  ros::Time capture_time = ros::Time::now();

  // Stereo matching task
  NxLibCommand computeDisparityMap(cmdComputeDisparityMap);
  computeDisparityMap.parameters()[itmCameras] = (*pCamera_)[itmSerialNumber].asString();
  computeDisparityMap.execute();

  // Convert disparity map into XYZ data for each pixel
  NxLibCommand computePointMap(cmdComputePointMap);
  computePointMap.parameters()[itmCameras] = (*pCamera_)[itmSerialNumber].asString();
  computePointMap.execute();
  std::cout << "End stereo processing" << std::endl;

  return (true);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::extractPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl)
{
  // Get info about the computed point map and copy it into a std::vector
  std::vector<float> pointMap;
  int width, height;
  (*pCamera_)[itmImages][itmPointMap].getBinaryDataInfo(&width, &height, 0, 0, 0, 0);
  (*pCamera_)[itmImages][itmPointMap].getBinaryData(pointMap, 0);

  // Copy point cloud and convert in meters
  pcl->points.resize(height * width);
  pcl->width = width;
  pcl->height = height;
  pcl->is_dense = false;

  // Copy data in point cloud (and convert millimeters to meters)
  for (size_t i = 0; i < pointMap.size(); i += 3) {
    pcl->points[i / 3].x = pointMap[i] / 1000.0;
    pcl->points[i / 3].y = pointMap[i + 1] / 1000.0;
    pcl->points[i / 3].z = pointMap[i + 2] / 1000.0;
  }
  return (true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::saveImage(int leftVsRight, int rawVsRectified, std::string filePath)
{

  NxLibCommand(cmdRectifyImages).execute();

  // Save images
  NxLibCommand saveImage(cmdSaveImage);
  if (leftVsRight == ENSENSO_CAMERA_2) {
    if (rawVsRectified == ENSENSO_CAMERA_RAW) {
      saveImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRaw][itmLeft].path;
    } else {
      saveImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRectified][itmLeft].path;
    }
  } else {
    if (rawVsRectified == ENSENSO_CAMERA_RAW) {
      saveImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRaw][itmRight].path;
    } else {
      saveImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRectified][itmRight].path;
    }
  }
  saveImage.parameters()[itmFilename] = filePath;
  saveImage.execute();

  return (true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoCameraInterface::loadImage(int leftVsRight, int rawVsRectified, std::string filePath)
{

  NxLibCommand loadImage(cmdLoadImage);
  if (leftVsRight == ENSENSO_CAMERA_2) {
    if (rawVsRectified == ENSENSO_CAMERA_RAW) {
      loadImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRaw][itmLeft].path;
    } else {
      loadImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRectified][itmLeft].path;
    }
  } else {
    if (rawVsRectified == ENSENSO_CAMERA_RAW) {
      loadImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRaw][itmRight].path;
    } else {
      loadImage.parameters()[itmNode] = (*pCamera_)[itmImages][itmRectified][itmRight].path;
    }
  }
  loadImage.parameters()[itmFilename] = filePath;
  try {
    loadImage.execute();
  }
  catch (NxLibException &ex) {
    ROS_ERROR("Failed to load camera image from this file: %s", filePath.c_str());
    ensensoExceptionHandling(ex, "loadImage");
    return (false);
  }

  return (true);
}

/***************************************************************************************************************/
/* CLASS EnsensoSystemInterface */
/***************************************************************************************************************/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoSystemInterface::EnsensoSystemInterface()
{
  ROS_INFO ("Initializing nxLib\n");
  try {
    nxLibInitialize ();
    pRoot_ = new NxLibItem;
  }
  catch (NxLibException &ex) {
    ensensoExceptionHandling (ex, "EnsensoSystemInterface");
    PCL_THROW_EXCEPTION (pcl::IOException, "Could not initialize NxLib."); // If constructor fails; throw exception
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EnsensoSystemInterface::~EnsensoSystemInterface()
{
  if (pRoot_)
    delete pRoot_;
  ROS_INFO ("Deleted pRoot \n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EnsensoSystemInterface::enumDevices(bool *allCamsAvailable)
{
  int i;


    NxLibItem cams = NxLibItem("/Cameras/BySerialNo");

	*allCamsAvailable = true;
	// Print information for all cameras in the tree
	ROS_INFO("\n");
	ROS_INFO("Number of connected cameras: %d", cams.count());
	ROS_INFO("Serial#      Model             EEPROM ID    Status");
	for (i = 0; i < cams.count(); i++) {
	  ROS_INFO("%s      %s      %d       %s", cams[i][itmSerialNumber].asString().c_str(), cams[i][itmModelName].asString().c_str(),
			  cams[i][itmEepromId].asInt(), cams[i][itmStatus].asString().c_str());
	  if (cams[i][itmStatus].asString().compare("Available")) {
          *allCamsAvailable = false;
	  } else {
	  }
	}
	/* If all of the cameras are available to use, open them all */
	if (*allCamsAvailable) {
		for (i = 0; i < cams.count(); i++) {
		  EnsensoCamPtr cam(new EnsensoCameraInterface);
		  // Add a new camera object to the camera list
		  ensensoCams.push_back(cam);
		  ensensoCams[i]->device_num_ = i;
		  // Create a pointer referencing the camera's tree item:
		  *(ensensoCams[i]->pCamera_) = (*pRoot_)[itmCameras][itmBySerialNo][i];
		}
	}
	ROS_INFO("\n");

  return (ensensoCams.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EnsensoSystemInterface::enumDevicesSim(int numSensors)
{
  int i;

  // Print information for all cameras in the tree
  ROS_INFO("Simulation using 2 cameras.");

  for (i = 0; i < numSensors; i++) {
    EnsensoCamPtr cam(new EnsensoCameraInterface);
    // Add a new camera object to the camera list
    ensensoCams.push_back(cam);
    ensensoCams[i]->device_num_ = i;
    // Create a pointer referencing the camera's tree item:
    *(ensensoCams[i]->pCamera_) = (*pRoot_)[itmCameras][itmBySerialNo][i];
  }

  return (ensensoCams.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::openAllDevices(void)
{
  int i;

  for (i = 0; i < ensensoCams.size(); i++) {
    if (!ensensoCams[i]->openDevice()) {
      //ui_report_status(gyges_status::ENSENSOS_FAILED_TO_OPEN);
      ROS_ERROR("Failed to open Ensenso device %d.  Aborting.", i);
      return (false);
    } else {
      ROS_INFO("Opened Ensenso device %d", i);
    }
  }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::closeAllDevices(void)
{
  int i;

  for (i = 0; i < ensensoCams.size(); i++) {
    if (!ensensoCams[i]->closeDevice()) {
      ROS_ERROR("Failed to close Ensenso device %d.  Aborting.", i);
      return (false);
    } else {
      ROS_INFO("Closed Ensenso device %d", i);
    }
  }
  return (true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::triggerSingleSensor(int ensenso_eeprom_id)
{
	int ensensoIdx = findEepromIdIndex(ensenso_eeprom_id);

	ROS_DEBUG("Started capture");
	NxLibCommand capture(cmdCapture);
	capture.parameters() [itmCameras] = (*(ensensoCams[ensensoIdx]->pCamera_))[itmSerialNumber].asString();
	capture.execute();

	// Stereo matching task
	ROS_DEBUG("Started stereo processing");
	NxLibCommand computeDisparityMap(cmdComputeDisparityMap);
	computeDisparityMap.parameters() [itmCameras] = (*(ensensoCams[ensensoIdx]->pCamera_))[itmSerialNumber].asString();
	computeDisparityMap.execute ();

	// Convert disparity map into XYZ data for each pixel
	NxLibCommand computePointMap(cmdComputePointMap);
	computePointMap.parameters() [itmCameras] = (*(ensensoCams[ensensoIdx]->pCamera_))[itmSerialNumber].asString();
	computePointMap.execute ();
  ROS_DEBUG("Finished stereo processing");

    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::triggerAllSensors(void)
{
  ROS_DEBUG("Started capture");
  NxLibCommand(cmdCapture).execute();

  ROS_DEBUG("Started stereo processing");
  // Stereo matching task
  NxLibCommand(cmdComputeDisparityMap).execute();

  // Convert disparity map into XYZ data for each pixel
  NxLibCommand(cmdComputePointMap).execute();

  ROS_DEBUG("Finished stereo processing");
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::triggerAllSensorsHWSim(std::string simDirPath)
{
  std::string fileNameFound;

  ROS_INFO("Loading simulation data");
  // Load system configuration file
  getFileNameMatch("jsontree.json", simDirPath, &fileNameFound);
  configureSystemFromJson(fileNameFound);

  ROS_INFO("Started stereo processing (simulation mode)");
  // Load the raw images
  loadAllImages(simDirPath);

  // Stereo matching task
  NxLibCommand(cmdComputeDisparityMap).execute();

  // Convert disparity map into XYZ data for each pixel
  NxLibCommand(cmdComputePointMap).execute();
  ROS_INFO("Finished stereo processing (simulation mode)");
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::triggerAllSensorsSWSim(std::string simDirPath)
{
  std::string pclFileName, pclFileNameFound;
  pcl::PCDReader pcl_reader;

  // Read the aggregate point cloud file from the simulation directory
  getFileNameMatch(".pcd", simDirPath, &pclFileNameFound);
  pcl_reader.read(pclFileNameFound, aggregate_point_cloud_);

  ROS_INFO("Aggregate point cloud file read.  ");
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::generateAggregatePointCloud(int eeprom_id)
{
  int i;
  int startIdx, endIdx;
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > extractedPcl(new pcl::PointCloud<pcl::PointXYZ>);

  int ensensoIdx = findEepromIdIndex(eeprom_id);

  /* Choose which camera point clouds to include in the aggregate point cloud */
  if (ensensoIdx == ALL_ENSENSOS) {
	  startIdx = 0;
	  endIdx = ensensoCams.size();
  } else {
	  startIdx = ensensoIdx;
	  endIdx = ensensoIdx + 1;
  }

  ROS_DEBUG("Generating aggregate point cloud");
  aggregate_point_cloud_.clear();
  /* Combine all of the individual point clouds into a single aggregate point cloud */
  for (i = startIdx; i < endIdx; i++) {
    ensensoCams[i]->extractPointCloud(extractedPcl);
    aggregate_point_cloud_ += *extractedPcl;
  }

  return (true);
}

void getFileNameMatch(std::string searchString, std::string dirPath, std::string *filePathFound)
{
  namespace fs = ::boost::filesystem;
  std::string tmpStr;
  std::size_t foundPos;
  bool found = false;

  *filePathFound = "";

  if (!fs::exists(dirPath) || !fs::is_directory(dirPath))
    return;

  fs::recursive_directory_iterator it(dirPath);
  fs::recursive_directory_iterator endit;
  while ((it != endit) && !found) {
    tmpStr = it->path().string();
    foundPos = tmpStr.find(searchString);
    if (foundPos != std::string::npos) {
      *filePathFound = tmpStr;
      found = true;
    }
    ++it;
  }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::loadAllImages(std::string dirPath)
{
  int i;
  std::string fileNameFound;

  /* Load in raw images for each of the cameras */
  for (i = 0; i < ensensoCams.size(); i++) {
    int eeprom_id;
    std::stringstream ss;

      /* Add the camera serial number into the image name */
    eeprom_id = (*(ensensoCams[i]->pCamera_))[itmEepromId].asInt();
    ss << eeprom_id;
    getFileNameMatch("EEPROM_ID_" + ss.str() + "_leftRaw", dirPath, &fileNameFound);
    ensensoCams[i]->loadImage(ENSENSO_CAMERA_2, ENSENSO_CAMERA_RAW, fileNameFound);
    getFileNameMatch(ss.str() + "_rightRaw", dirPath, &fileNameFound);
    ensensoCams[i]->loadImage(ENSENSO_CAMERA_1, ENSENSO_CAMERA_RAW, fileNameFound);
  }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::logImages(int eeprom_id, std::string dirPath)
{
  int i, startIdx, endIdx;
  int ensensoIdx = findEepromIdIndex(eeprom_id);

  if (ensensoIdx == ALL_ENSENSOS)
  {
    startIdx = 0;
    endIdx = ensensoCams.size();
  } else
  {
    startIdx = ensensoIdx;
    endIdx = ensensoIdx + 1;
  }

  /* Save both raw and rectified images for each of the cameras */
  for (i = startIdx; i < endIdx; i++)
  {
    int eeprom_id;
    std::stringstream ss;

    /* Add the camera serial number into the image name */
    eeprom_id = (*(ensensoCams[i]->pCamera_))[itmEepromId].asInt();
    ss << eeprom_id;

    ensensoCams[i]->saveImage(ENSENSO_CAMERA_2, ENSENSO_CAMERA_RAW,
        dirPath + "//EEPROM_ID_" + ss.str() + "_leftRaw.png");
    ensensoCams[i]->saveImage(ENSENSO_CAMERA_1, ENSENSO_CAMERA_RAW,
        dirPath + "//EEPROM_ID_" + ss.str() + "_rightRaw.png");
    ensensoCams[i]->saveImage(ENSENSO_CAMERA_2, ENSENSO_CAMERA_RECTIFIED,
        dirPath + "//EEPROM_ID_" + ss.str() + "_leftRectified.png");
    ensensoCams[i]->saveImage(ENSENSO_CAMERA_1, ENSENSO_CAMERA_RECTIFIED,
        dirPath + "//EEPROM_ID_" + ss.str() + "_rightRectified.png");
  }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::logAggregatePointCloud(std::string dirPath)
{
  pcl::PCDWriter pcl_writer;
  std::vector<int> indices;
  pcl::PointCloud<pcl::PointXYZ> aggregate_point_cloud_cleaned;

  pcl::removeNaNFromPointCloud(aggregate_point_cloud_, aggregate_point_cloud_cleaned, indices);
  pcl_writer.writeBinaryCompressed(dirPath + "//" + "pcl.pcd", aggregate_point_cloud_cleaned);
  return (true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoSystemInterface::configureSystemFromJson(std::string jsonFilePath)
{
  bool pretty_format = true;
  std::ifstream inFile;
  std::stringstream jsonTreeStr;

  inFile.open(jsonFilePath.c_str());
  if (!inFile.is_open()) {
    ROS_ERROR("Failed to open json configuration file in EnsensoSystemInterface::configureCameraFromJson");
    ROS_ERROR("Json file name: %s", jsonFilePath.c_str());
    return (false);
  } else {
    /* Read the entire json file contents into a string */
    jsonTreeStr << inFile.rdbuf();

    try {
      /* Put the json string contents into the NxLib tree */
      (*pRoot_).setJson(jsonTreeStr.str(), NXLIBTRUE);
    }
    catch (NxLibException &ex) {
      ensensoExceptionHandling(ex, "configureSystemFromJson");
      return (false);
    }
    ROS_INFO("Ensenso System configured.");
    return (true);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::logNxLibTreeAsJson(std::string dirPath)
{
  bool pretty_format = true;
  std::string jsonTreeStr;
  std::string fileNameComplete;

  try {
    jsonTreeStr = (pRoot_->asJson(pretty_format));
  }
  catch (NxLibException &ex) {
    ensensoExceptionHandling(ex, "getTreeAsJson");
    jsonTreeStr = "";
    ROS_ERROR("Failed to retrieve json string in EnsensoSystemInterface::saveNxLibTreeAsJson");
    return (false);
  }

  // Write the json string to a file
  fileNameComplete = dirPath + "//" + "jsontree.json";
  std::ofstream out(fileNameComplete.c_str());
  if (!out.is_open()) {
    ROS_ERROR("Failed to create json log file in EnsensoSystemInterface::saveNxLibTreeAsJson");
    return (false);
  } else {
    out << jsonTreeStr << std::endl;
    out.flush();
    out.close();
  }
  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EnsensoSystemInterface::logAll(int eeprom_id, std::string dirPath, std::string logTag, int activity_id)
{
//  std::string timeStampStr;
//  std::stringstream ss_s, ss_ns;
//  ros::Time capture_time;
//  ros::Time log_time;
//  boost::filesystem::path newLogDirName;
//  std::stringstream ss;
//
//  std::string hostname;
//  // TODO: Change to appropriate values
//  std::string eventstr = "PrePick";
//  ss << eeprom_id;
//  std::string cameraID = ss.str();
//  std::string imageType = "image_type";
//  std::string fileFormat = "png";
//  ensenso::GetDataBaseHostName(hostname);
//
//  log_time = ros::Time::now();
//  ss_s << log_time.sec;
//  ss_ns << log_time.nsec;
//  timeStampStr = ss_s.str() + "." + ss_ns.str();
//
//  newLogDirName = dirPath + "/" + logTag + timeStampStr;
//  if (!boost::filesystem::create_directory(newLogDirName)) {
//    ROS_ERROR("Failed to new log directory!");
//  } else {
//    // Save the various data of interest: image files, point cloud data, and json configuration
//    logImages(eeprom_id, newLogDirName.string());
//    logAggregatePointCloud(newLogDirName.string());
//    logNxLibTreeAsJson(newLogDirName.string());
//
//    // Store the logs to the NAS as well
//    ensenso::Move_Photos(hostname, activity_id, eventstr, cameraID, imageType, fileFormat, newLogDirName.string());
//
//  }
  return (true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
EnsensoSystemInterface::configureAllCamerasFromJsons(std::string jsonDirPath)
{
  int i;
  int camSerial;
  std::string filePathComplete;
  std::string tempStr;

  for (i = 0; i < ensensoCams.size(); i++) {
    std::stringstream ss;

    camSerial = (*(ensensoCams[i]->pCamera_))[itmEepromId].asInt();
    ss << camSerial;
    filePathComplete = jsonDirPath + "/" + ss.str() + ".json";
    tempStr = "Opening Ensenso json config file:" + filePathComplete;
    ROS_INFO("%s", tempStr.c_str());
    ensensoCams[i]->configureCameraFromJson(filePathComplete);
  }

  return (true);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
EnsensoSystemInterface::findEepromIdIndex(int eepromId)
{
  int i;
  int matchIdx = -1;
  int eeprom_id_current;

  for (i = 0; i < ensensoCams.size(); i++) {
	  eeprom_id_current = (*(ensensoCams[i]->pCamera_))[itmEepromId].asInt();
	  if (eeprom_id_current == eepromId) {
		  matchIdx = i;
	  }
  }
  if (matchIdx < 0) {
	  ROS_ERROR("Ensenso EEPROM index not found");
  }

  return matchIdx;
}
