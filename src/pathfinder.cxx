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
#ifndef __ENSENSO_HEADERS_H__
#define __ENSENSO_HEADERS_H__
#endif

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
