#include <tuple>
#include <boost/filesystem.hpp>

namespace pathfinder
{
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
