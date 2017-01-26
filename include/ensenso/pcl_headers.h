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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>


/*Globlal namespaces and aliases*/
using PointT 			= pcl::PointXYZ;
using PointCloudT 		= pcl::PointCloud<PointT>;
using PointCloudTPtr 	= PointCloudT::Ptr;

using PointN  			= pcl::Normal;
using PointCloudN 		= pcl::PointCloud<PointN>;
using PointCloudNPtr  	= PointCloudN::Ptr;

using pcl_viz 			= pcl::visualization::PCLVisualizer;
using NormalEstimation 	= pcl::NormalEstimation<PointT, PointN>;

using CRH90 				= pcl::Histogram<90>;
using PointCloudCRH90		= pcl::PointCloud<CRH90>;
using PointCloudCRH90Ptr	= PointCloudCRH90::Ptr;

/*Descriptor aliases*/
using PFH125 				= pcl::PFHSignature125;
using PointCloudPFH125 		= pcl::PointCloud<PFH125>;
using PointCloudPFH125Ptr 	= pcl::PointCloud<PFH125>::Ptr;

/*Kd Trees*/
using TreeKd = pcl::search::KdTree<PointT>;
using TreeKdPtr = pcl::search::KdTree<PointT>::Ptr;

using VFH308 = pcl::VFHSignature308;
using PointCloudVFH308 = pcl::PointCloud<VFH308>;
using PointCloudTVFH308Ptr = PointCloudVFH308::Ptr;

#define OUT(__o__) std::cout<< __o__ << std::endl;