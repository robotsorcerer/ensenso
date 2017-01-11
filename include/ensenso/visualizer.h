#ifndef _VISUALIZER_H_
#define _VISUALIZER_H_

#define SENSOR_IP			    "192.168.1.10"			

#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/common_headers.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <boost/thread/thread.hpp>

#include <sstream>
#include <string>
#include <vector>

pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;

/*Custom typedefs*/
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using CloudViewer = pcl::visualization::CloudViewer;

/*Globals*/

class visualizer
{
private:
	double pos_x, pos_y, pos_z, view_x;
	double view_y, view_z, view_up_x, view_up_y, view_up_z;
	unsigned int text_id, viewport;
	bool updateCloud, save;	

	const int screen_height, screen_width;
	const std::string pcl_viewer;

	pcl::PCDWriter writer;
public:
	//constructor
	visualizer()
	: pos_x(0), pos_y(0), pos_z(0), view_x(0), 
		view_y(-1), view_z(0), view_up_x(0), view_up_y(1), view_up_z(1), text_id(0),
		viewport(0), updateCloud(false), save(false), screen_height(640), 
		screen_width(480), pcl_viewer("ensenso cloud")
	{
		/*Use std::initializer_liust ctro; vector has 2-elements*/
	}

	//Destructor
	~visualizer()
	{		}

	void start()
	{	
		std::cout <<"\n\n";
		ROS_INFO("=================================================================");
		ROS_INFO("                                                                 ");
		ROS_INFO("        Code by Olalekan Ogunmolu <<ogunmolu@amazon.com>>        ");
		ROS_INFO("                                                                 ");
		ROS_INFO("                  Press 'q'  on cloud to quit                    ");
		ROS_INFO("=================================================================");
		std::cout << "\n\n";
	}

	pcl::PCDWriter getPCDWriter() const
	{
		return writer;
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> createViewer()
	{

	 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL Viewer"));
		
	 	viewer->initCameraParameters();
	 	viewer->setBackgroundColor(128/255, 128/255, 128/255);
	 	viewer->setSize(screen_height, screen_width);
	 	viewer->setShowFPS(true);
	 	viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z) ;	
	 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, \
	 		                                      50.0,/* 13.0, 13.0,*/ pcl_viewer, viewport);
	 	// viewer->addCoordinateSystem(0.5);
	 	viewer->registerKeyboardCallback(&visualizer::keyboardEventOccurred, *this);	 	

	  return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> colorViewer()
	{
	  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Colored Viewer"));
	  viewer->setBackgroundColor (0, 0, 0);
	  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, pcl_viewer);
	  viewer->setSize(screen_height, screen_width);
	  viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z) ;	
	  viewer->setShowFPS(true);
	  viewer->initCameraParameters ();
	  viewer->addCoordinateSystem (1.0);
	 	viewer->registerKeyboardCallback(&visualizer::keyboardEventOccurred, *this);
	  return (viewer);
	}


	void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
	                            void* )
	{
	  if (event.keyUp())
	  {
	  	switch(event.getKeyCode())
	  	{
	  		case 'q':
	  		  // this->quit();
	  		  break;
	  		case ' ':	
	  		case 's':
	  			save=true;
	  			// ROS_INFO_STREAM("key code for s: " << event.getKeyCode());
	  		  break;
	  	}
	  }
	}

	void mouseEventOccurred(const pcl::visualization::MouseEvent &event,
	                         void* viewer_void)
	{
	  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
	      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
	  {
	    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

	    char str[512];
	    sprintf (str, "text#%03d", text_id ++);
	    viewer->addText ("clicked here", event.getX (), event.getY (), str);
	  }
	}
};

#endif