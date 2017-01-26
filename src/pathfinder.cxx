#include <tuple>
#include <boost/foreach.hpp>
/*
Without defining boost scoped enums, linking breaks the compilation. 
See http://stackoverflow.com/questions/15634114/cant-link-program-using-boost-filesystem
*/
#define BOOST_NO_CXX11_SCOPED_ENUMS		
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/tuple/tuple.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <ros/package.h>  /*roslib*/
#include <ensenso/ensenso_headers.h>
#include <Eigen/Dense>

namespace pathfinder
{
	static bool getROSPackagePath(const std::string pkgName, boost::filesystem::path & pkgPath)
	{
	    pkgPath = ros::package::getPath(pkgName);
	    if (pkgPath.empty())
	    {
	        printf("Could not find package '%s' ", pkgName.c_str());
	        return false;
	    }
	    else
	    {
	        printf("%s package found here: %s", pkgName.c_str(), pkgPath.c_str());
	        return true;
	    }
	}

	static bool copyDirectory(const boost::filesystem::path srcPath, const boost::filesystem::path dstPath)
	{
	    boost::filesystem::create_directory(dstPath);

	    for (boost::filesystem::directory_iterator end, dir(srcPath.c_str()); dir != end; ++dir)
	    {
	        boost::filesystem::path fn = (*dir).path().filename();
	        boost::filesystem::path srcFile = (*dir).path();
	        //cout << "     Source file: " << srcFile.c_str() << endl;
	        boost::filesystem::path dstFile = dstPath / fn;
	        //cout << "Destination file: " << dstFile.c_str() << endl;
	        boost::filesystem::copy_file(srcFile, dstFile);
	    }
	    return true;
	}

	bool cloudsAndImagesPath(boost::filesystem::path & imagesPath, boost::filesystem::path & cloudsPath, const std::string& pkgName)
	{
		boost::filesystem::path pkgPath;
		getROSPackagePath(pkgName, pkgPath);
		ROS_INFO("Ensenso Path %s", pkgPath.c_str());
		cloudsPath = pkgPath / "clouds";
		imagesPath = pkgPath / "images";
		if(!imagesPath.empty() && !cloudsPath.empty())
		{
			ROS_INFO("images path: %s \n clouds path: %s", imagesPath.c_str(), cloudsPath.c_str());
			return true;
		}
		else
			ROS_INFO("paths %s, and %s not found", imagesPath.c_str(), cloudsPath.c_str());
	}

/*	void getClouds(boost::filesystem::path & path, std::vector<PointCloudTPtr>pclFiles)
	{
		for (boost::filesystem::directory_iterator it (path); it != boost::filesystem::directory_iterator (); ++it)
		{
		  if (boost::filesystem::is_directory (it->status ()))
		  {
		    std::stringstream ss;
		    ss << it->path ();
		    ROS_INFO ("Loading %s ", ss.str ().c_str ());
			pcl::io::loadPCDFile<PointT>(cloudsPath.c_str() +  "/background_5_cloud.pcd", *cloud_background);
		  }
		  if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension)
		  {
		    vfh_model m;
		    if (loadHist (base_dir / it->path ().filename (), m))
		      models.push_back (m);
		  }
		}
	}*/

	std::tuple<boost::filesystem::path, const std::string&, const std::string&,\
				const std::string&, const std::string&, const std::string&, \
				const std::string&> getCurrentPath()
	{	
		boost::filesystem::path pwd = boost::filesystem::current_path();
		const std::string& train_dir = "train";
		const std::string& test_dir  = "test";
		const std::string& train_imgs = train_dir + "/images";
		const std::string& train_clouds = train_dir + "/clouds";
		const std::string& test_imgs = test_dir + "/images";
		const std::string& test_clouds = test_dir + "/clouds";

		/*Create the training directories*/
		if(!boost::filesystem::exists(train_dir))
		{
			boost::filesystem::create_directory(train_dir);
			if(!boost::filesystem::exists(train_imgs))
			{
				boost::filesystem::create_directory(train_imgs);			
			}
			if(!boost::filesystem::exists(train_clouds))
			{
				boost::filesystem::create_directory(train_clouds);			
			}
		}

		/*Create the testing directories*/
		if(!boost::filesystem::exists(test_dir))
		{
			boost::filesystem::create_directory(test_dir);
			if(!boost::filesystem::exists(test_imgs))
			{
				boost::filesystem::create_directory(test_imgs);			
			}
			if(!boost::filesystem::exists(test_clouds))
			{
				boost::filesystem::create_directory(test_clouds);			
			}
		}

		return std::make_tuple(pwd, train_dir, train_imgs, train_clouds, \
						test_dir, test_imgs, test_clouds);
	}
}


namespace cam_info{

	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getLeftCam()
	{
		Eigen::Matrix3d leftKMat, leftRotation;
		Eigen::Matrix4d leftProjection;
		leftKMat << 967.59025686735504, 0, 0,
						0,		967.59025686735504,		0,
				115.33532536063308,	394.56512026860639,	1;

		leftRotation << 	0.99783369813232914,
							0.0027184390085535194,
							-0.065730669865113958,
							-0.0027704131437143527,
							0.999995917693007,
							-0.0006995772535153315,
							0.065728499774245275,
							0.00088016286974392205,
							0.99783715586800525;
		leftProjection << 	1,0,0,0,
							0,	1,0,0,
							0,	0,0,0.0098904973412229247,
						    -115.33532536063308,-394.56512026860639,
						    967.59025686735504,1.7472966090675965;
		return std::make_tuple(leftKMat, leftRotation,leftProjection);

	}

	std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix4d>
				 getRightCam()
	{
		Eigen::Matrix3d rightKMat, rightRotation;
		Eigen::Matrix4d rightProjection;
		rightKMat << 967.59025686735504, 0, 0,
						0,		967.59025686735504,		0,
				115.33532536063308,	394.56512026860639,	1;

		rightRotation << 	0.99783369813232914,
							0.0027184390085535194,
							-0.065730669865113958,
							-0.0027704131437143527,
							0.999995917693007,
							-0.0006995772535153315,
							0.065728499774245275,
							0.00088016286974392205,
							0.99783715586800525;
		rightProjection << 	1,0,0,0,
							0,	1,0,0,
							0,	0,0,0.0098904973412229247,
						    -115.33532536063308,-394.56512026860639,
						    967.59025686735504,1.7472966090675965;
		return std::make_tuple(rightKMat, rightRotation,rightProjection);

	}
}