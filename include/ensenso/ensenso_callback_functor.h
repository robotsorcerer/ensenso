

#ifndef _ENSENSO_CB_FUNCTOR_
#define _ENSENSO_CB_FUNCTOR_

/**
 * \file ensenso_callback_functor.h
 * \brief helper class to easily allow user to write their own callback, without knowledge
 * of boost. 
 * \author lgalup
 *
 *
 * here is how you write a callback for the ensenso grabber
 * 1) write a new class which subclasses from ensenso_cb_functor
 * 2) implement your own execute method
 * 3) call registerCallback on the ensenso_grabber interface, passing in the output of get_Callback()
 *
 * for example:
 * SimpleOpenNIProcessor v; //SimpleOpenNiProcessor is YOUR class,whcih you have impl'd somewhere
 * ensenso::EnsensoGrabber interface;
 
 * interface.registerCallback(v.get_Callback());
 *
 * yes, its just that easy.
 */

#include <sensor_msgs/PointCloud2.h>
#include <boost/shared_ptr.hpp>
#include <iostream>


//forward decls
namespace pcl
{
  struct PointXYZ;

  template <typename T> class PointCloud;
}


namespace ensenso
{
  /**
   * \brief this is the function signature that ensenso_grabber is expecting
   * note the kind of point cloud. if we start getting an RGB one, need to extend this!
   *
   */
  typedef void
    (sig_cb_ensenso_point_cloud) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);

  typedef void
    (sig_cb_ensenso_point_cloud2) (const boost::shared_ptr< ::sensor_msgs::PointCloud2 >&);

  /**
   * \brief brief helper class for doing callbacks
   *
   *  up to the user to implement their own execute function
   */
  class ensenso_cb_functor
  {
  public:
    virtual
      ~ensenso_cb_functor() {};

    virtual
      void execute(const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&) = 0;


    boost::function<void (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&) >
      get_Callback()
      {
        return boost::bind (&ensenso_cb_functor::execute, this, _1);
      }
  
  };

  class ensenso_cb_functor2
  {
  public:
    virtual
      ~ensenso_cb_functor2() {};

    virtual
      void execute(const boost::shared_ptr< ::sensor_msgs::PointCloud2 >&) = 0;


    boost::function<void (const boost::shared_ptr< ::sensor_msgs::PointCloud2 >&) >
      get_Callback()
      {
        return boost::bind (&ensenso_cb_functor2::execute, this, _1);
      }
  
  };

}


#endif




