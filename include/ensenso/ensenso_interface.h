#ifndef __ENSENSO_INTERFACE__
#define __ENSENSO_INTERFACE__

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_config.h>
#include <pcl/common/time.h>
#include <pcl/io/eigen.h>
#include <pcl/io/boost.h>
#include <boost/shared_ptr.hpp>
// Ensenso SDK
#include <nxLib.h>

// Maximum number of allowed Ensenso cameras in this system
#define MAX_ENSENSO_CAMS	10

enum
{
  ALL_ENSENSOS = -1,
  ENSENSO_CAMERA_1 = 1,
  ENSENSO_CAMERA_2 = 2
};
enum
{
  ENSENSO_CAMERA_RAW, ENSENSO_CAMERA_RECTIFIED
};

namespace ensenso
{
/** @brief Interface for IDS-Imaging Enenso's devices
 */
class EnsensoCameraInterface
{
public:
  /** @brief Whether an Ensenso device is opened or not */
  int device_num_;

  /** @brief Reference to the camera tree */
  NxLibItem * pCamera_;

  /** @brief Constructor */
  EnsensoCameraInterface();

  /** @brief Destructor */
  ~EnsensoCameraInterface();

  /** @brief Opens the camera interface */
  bool
  openDevice(void);

  /** @brief Closes the camera interface */
  bool
  closeDevice(void);

  /** @brief Configure Ensenso capture settings
   * @param[in] auto_exposure If set to yes, the exposure parameter will be ignored
   * @param[in] auto_gain if set yo yes, the gain parameter will be ignored
   * @param[in] binning Pixel binning: 1, 2 or 4
   * @param[in] exposure In milliseconds, from 0.01 to 20 ms
   * @param[in] front_light Infrared front light (useful for calibration)
   * @param[in] gain Float between 1 and 4
   * @param[in] gain_boost
   * @param[in] hardware_gamma
   * @param[in] hdr High Dynamic Range (check compatibility with other options in Ensenso manual)
   * @param[in] pixel_clock in MegaHertz, from 5 to 85
   * @param[in] projector Use the central infrared projector or not
   * @param[in] target_brightness Between 40 and 210
   * @param[in] trigger_mode
   * @param[in] use_disparity_map_area_of_interest
   * @return True if successful, false otherwise
   * @note See <a href="http://www.ensenso.de/manual/index.html?capture.htm">Capture tree item</a> for more
   * details about the parameters. */
  bool
  configureCapture(const bool auto_exposure = true, const bool auto_gain = true, const int binning = 1,
                   const float exposure = 0.32, const bool front_light = false, const int gain = 1,
                   const bool gain_boost = false, const bool hardware_gamma = false, const bool hdr = false,
                   const int pixel_clock = 10, const bool projector = true, const int target_brightness = 80,
                   const std::string trigger_mode = "Software",
                   const bool use_disparity_map_area_of_interest = false) const;

  /** @brief Configure Ensenso disparity map settings */
  bool
  configureCameraFromJson(std::string jsonFilePath);

  /** @brief Initiates captures point cloud data on a single camera interface */
  bool
  triggerSensor(void);

  /** @brief Returns point cloud data for this interface after a capture
   * has been initiated */
  bool
  extractPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl);

  /** @brief Saves an image from the specified side of the Ensenso to the given
   * filePath  */
  bool
  saveImage(int leftVsRight, int rawVsRectified, std::string filePath);
  /** @brief Loads an image from the file to the specified side of the Ensenso */
  bool
  loadImage(int leftVsRight, int rawVsRectified, std::string filePath);
protected:
  /** @brief Whether an Ensenso device is opened or not */
  bool device_open_;

};
typedef boost::shared_ptr<EnsensoCameraInterface> EnsensoCamPtr;

/** @brief Interface for IDS-Imaging Enenso's devices
 */
class EnsensoSystemInterface
{
public:
  /** @brief Reference to the NxLib tree root
   * @warning You must handle NxLib exceptions manually when playing with root!
   * See void ensensoExceptionHandling in ensenso_grabber.cpp */
  NxLibItem * pRoot_;

  /** @brief Storage for aggregate point clouds cloud (merge of all point clouds) */
  pcl::PointCloud<pcl::PointXYZ> aggregate_point_cloud_;

  /** @brief Vector of pointers to cameras in the system */
  std::vector<EnsensoCamPtr> ensensoCams;

  /** @brief Constructor */
  EnsensoSystemInterface();

  /** @brief Destructor  */
  ~EnsensoSystemInterface();

  /** @brief Searches for available devices
   * @returns the number of Ensenso devices connected */
  int
  enumDevices(bool *allCamsAvailable);
  int
  enumDevicesSim(int numSensors);

  /** @brief Opens all Ensenso devices available in the system */
  bool
  openAllDevices(void);

  /** @brief Closes all Ensenso devices available in the system */
  bool
  closeAllDevices(void);

  /** @brief Initiates capture of point cloud data on all camera interfaces */
  bool
  triggerSingleSensor(int ensensoIdx);

  /** @brief Initiates capture of point cloud data on all camera interfaces */
  bool
  triggerAllSensors(void);
  bool
  triggerAllSensorsHWSim(std::string simDirPath);
  bool
  triggerAllSensorsSWSim(std::string simDirPath);

  /** @brief Generate aggregate point cloud from separate camera point clouds */
  bool
  generateAggregatePointCloud(int camEEpromIdxToUse);

  /** @brief Finds and loads camera images from the specified directory into the
   * nxLib tree */
  bool
  loadAllImages(std::string dirPath);

  /** @brief Save all of the sensor images (left, right, raw, and rectified)
   * in the specified directory for the specified ensensoIdx.  ensensoIdx = -1
   * logs images for all sensors.  */
  bool
  logImages(int ensensoIdx, std::string dirPath);

  /** @brief Saves aggregate point cloud in the specified directory */
  bool
  logAggregatePointCloud(std::string dirPath);

  /** @brief Saves the NxLib tree in the specified directory in json format */
  bool
  logNxLibTreeAsJson(std::string dirPath);

  /** @brief Create a directory at given location and create log files of all data from the
   * most recent system capture */
  bool
  logAll(int ensensoIdx, std::string dirPath, std::string logTag, int activity_id);

  /** @brief Configures all of the cameras from their saved json files */
  bool
  configureAllCamerasFromJsons(std::string jsonDirPath);

  /** @brief Configures entire Ensenso system from saved json file */
  bool
  configureSystemFromJson(std::string jsonFilePath);

  /** @brief Find the array index of the camera object with the given
   * EEPROM ID */
  int
  findEepromIdIndex(int eepromId);

protected:

};
}

#endif // __GANESHA_ENSENSO_INTERFACE__

