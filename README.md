
### ROS Bridge to the Ensenso SDK 
##### Drivers for retrieving point clouds from the ensenso camera

#### Author: [Olalekan Ogunmolu](http://lakehanne.github.io)

####[Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Dependencies](#dependencies)
	-[Optional Dependencies](#optional-dependencies)
- [Network Configuration](#network-configuration)
- [Using Ensenso on Linux](#using-ensenso-on-linux)
- [@TODOs](#@todos)
- [FAQs](#faqs)

#### Introduction

The sensor captures a single 3D profile for each camera exposure. The sensor projects a laser line to the target and the sensor's camera views the laser from an angle capturing the reflection of the light off the target. Due to the triangulation angle, the laser line appears in different positions on the camera depending on the 3D shape of the target.

#### Dependencies

* uEye Driver

  Please make sure you have the IDS uEye driver installed in version 4.00 or higher. For N20, N30 and N35 cameras a driver with version 4.41 or higher is required. Please visit [ensenso download](www.ensenso.com/download) page and download and install the latest driver for your system.

* EnsensoSDK

  After the uEye driver installation is complete, download the latest EnsensoSDK software release from [ensenso download](www.ensenso.com/download) and follow the instructions on screen to install the software.

*   PCL (Point Clouds Library)

  The sensor requires pcl-1.8.0.  Clone the 1.8.0 trunk from the [pcl github repo](https://github.com/PointCloudLibrary/pcl/tree/pcl-1.8.0), then follow the readme instructions to install. A good place to install might be in your Documents directory. After installing, be sure to set the directory of the cmake pcl config file to that where the libraries are installed. On my computer, the default is `/usr/local/share/pcl-1.8/PCLConfig.cmake` . Next, go to the CMakeLists.txt file within the root directory of the project and amend the PCL directory to the one where your installer puts it (you can easily find this in the installer_manifest.txt file of your build directory).

##### Optional Dependencies

  *	 VTK LIBRARY (Optional)
  *	 Boost Library (Optional)

### Network Configuration

The factory ip address has been changed to 192.168.1.11. On your system's ethernet configuration settings, you would want to set an ip address and subnet similar to the following:

```bash
ip address: 192.168.1.10
subnet: 255.255.255.0
gateway: [Leave blank]
```

Below is a visual description of a typical setup

<div class="fig figcenter fighighlight">
	<img src="/images/sys_network.png" height="250" width="49%" align="middle" >
	<img src="/images/ensenso_conf.png" height="250"  width="49%" align="right" style="border-left: 1px solid black;">
	</br>
	<div class="figcaption" align="middle" hspace="80"></div>
</div>

In addition, ensure your ubuntu firewall is turned off 

```bash
 sudo ufw disable
```

If you encounter problems during set-up, it might be worth the while runing the `ueyesetupid` executable packed into the gunzipped tarball available at [setup id](http://ecs.utdallas.edu/~opo140030/sensors/uEye-Linux-4.81-64-bit.tgz). You would want to unzip the tar ball and run the `ueyesdk-setup-4.81-eth-amd64.gz.run` file. This should install configuration files to your `/usr/local/share/ueye/bin` directory from where you can configure the id of your sensor, which should be 1 by default. 

#### Using Ensenso on Linux

1.	Power up the sensor. Connect the Power/LAN cable to the sensor and the other end to the Power and Ethernet ports on the RJ45 connector on your computer

2. Clone this repo to your catkin src folder then `catkin build` from your catkin workspace root.

3. All things being equal, you should be presented with a pcl window that stream the frames that are grabbed to your pcl cloud viewer


#### @TODOs
 =====
 Lekan 
 =====

 *	Segment profile map of head and find rpy and cartesian coordinates of head pose

 *  Publish clouds to rviz for onwards visualization

 ====
 Bashir
 ====
 Create your own branch off the master fork and process your clouds.

#### FAQS

##### I am having issues connecting to the sensor even though my code compiles

Be sure the id of the sensor is properly set using the `ids camera manager` configuration utility that ships with [ueye](http://ecs.utdallas.edu/~opo140030/sensors/uEye-Linux-4.81-64-bit.tgz). You can find this in the images folder of this package.

### I am having issues setting up a dual GigEthernet and wireless connection 

Please follow the advice listed in the Installation section

### Other queries
If you run into further issues, feel free to open an issues ticket or ping me [@patmeansnoble](https://twitter.com/patmeansnoble) on twitter.