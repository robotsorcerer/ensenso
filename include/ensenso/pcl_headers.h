/*Pcl io's*/
#include <pcl/io/pcd_io.h>
#include <pcl/io/ensenso_grabber.h>

/*pcl points */
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl_ros/point_cloud.h>

/*pcl commons*/
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

/*pcl visualizations*/
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>

/*pcl conversions*/
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

/*pcl segmentations*/
#include <pcl/features/crh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/pfh.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/normal_3d.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/method_types.h>

/*Filters, Indices*/
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>

