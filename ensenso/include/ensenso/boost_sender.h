#include <boost/asio.hpp>
#include "boost/bind.hpp"
#include <geometry_msgs/Pose.h>
#include "ensenso/ValveControl.h" //local msg communicator of control torques
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include <Eigen/Core>
/*@brief sends the pose info on the LAN/WAN to 
* every listener on the network that subscribes to the pose of the head
*
*/
namespace udp
{
	class sender
	{
	public:
		// constructor pose only
		sender(boost::asio::io_service& io_service,
		        const boost::asio::ip::address& multicast_address, 
		        const geometry_msgs::Pose& pose_info);

		// constructor control law only
		sender(boost::asio::io_service& io_service,
		        const boost::asio::ip::address& multicast_address, 
		        const ensenso::ValveControl& u_valves);

		//constructor control law + ref + pose
		sender(boost::asio::io_service& io_service,
		        const boost::asio::ip::address& multicast_address, 
		        const ensenso::ValveControl& u_valves, 
		        const Eigen::Vector3d& ref, 
		        const geometry_msgs::Pose& pose_info);

		// sender functions
		void handle_send_to(const boost::system::error_code& error);
		void handle_timeout(const boost::system::error_code& error);
	private:
	  boost::asio::ip::udp::endpoint endpoint_;
	  boost::asio::ip::udp::socket socket_;
	  boost::asio::deadline_timer timer_;
	  std::ostringstream os, osc, osall;
	  ensenso::ValveControl u_valves_;
	  geometry_msgs::Pose pose;
	  Eigen::Vector3d ref_;
	  std::string message_;
	};
}