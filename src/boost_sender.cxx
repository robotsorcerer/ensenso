#include <ensenso/boost_sender.h>
#include <ros/ros.h>

const short multicast_port = 30001;
const int max_message_count = 300;                //because my head can never be more than 2m above the ground :)


/*@brief sends the calculated pose info on the LAN/WAN to 
* every listener on the network that subscribes to the pose of the head
*
*/
using namespace udp;
//constructor
sender::sender(boost::asio::io_service& io_service,
    const boost::asio::ip::address& multicast_address, 
    const ensenso::HeadPose& pose_info)
  : endpoint_(multicast_address, multicast_port),
    socket_(io_service, endpoint_.protocol()),
    timer_(io_service), pose(pose_info)
{
  os << std::fixed << std::setfill ('0') << std::setprecision (6) << 
        pose.stamp << ", " << pose.seq << ", " << pose.x << ", " <<  
        pose.y << ", " << pose.z <<  ", " <<  pose.pitch << ", " << 
        pose.yaw;

  message_ = os.str();

  socket_.async_send_to(
      boost::asio::buffer(message_), endpoint_,
      boost::bind(&sender::handle_send_to, this,
        boost::asio::placeholders::error));
}

void sender::handle_send_to(const boost::system::error_code& error)
{
  if (!error && pose.z < max_message_count)
  {
    timer_.expires_from_now(boost::posix_time::seconds(1));
    timer_.async_wait(
        boost::bind(&sender::handle_timeout, this,
          boost::asio::placeholders::error));
  }
}

void sender::handle_timeout(const boost::system::error_code& error)
{
  if (!error )
  {
    std::ostringstream os;

    os << std::fixed << std::setfill ('0') << std::setprecision (6) << 
          pose.stamp << ", " << pose.seq << ", " << pose.x << ", " <<  
          pose.y << ", " << pose.z <<  ", " <<  pose.pitch << ", " << 
          pose.yaw;

    message_ = os.str();

    ROS_WARN("Message Timed Out. Please look into your send::handle_timeout function");

    socket_.async_send_to(
        boost::asio::buffer(message_), endpoint_,
        boost::bind(&sender::handle_send_to, this,
          boost::asio::placeholders::error));
  }
}    

