#include <vector>
#include <string>
#include <ensenso/ensenso_headers.h>
#include <ensenso_detect/window_manager.h>

using namespace std;
using namespace cv;

class ImagesLoader
{
public:
	//constructor
    ImagesLoader();
    //destructor
    ~ImagesLoader();

    bool get_images_path(boost::filesystem::path && facePosPath);
    bool get_files();
    std::vector<std::string> filename_abs, filename_full;
private:
	friend class CoordinatesPicker;
    boost::filesystem::path detectPath;
    bool verbose;
};

class CoordinatesPicker
{
public:
	CoordinatesPicker();

	~CoordinatesPicker();

public:
	//class public methods
	enum { NOT_READY = 0, IN_VOGUE = 1, READY = 2};

	static const int radius = 2;
	static const int thickness = -1;
  cv::Mat draw_img;
  boost::shared_ptr<WindowManager> wm_;

	void reset();
	void setImageAndWinName( Mat&& _image, string && _winName );
	void showImage() const;
	// void onMouse(int event, int x, int y, int flags, void* param);
	void mouseClick(int event, int x, int y, int flags, void* );
	void nextIter();
	int getCount() const;
	void get_points(cv::Mat&& img, std::string&& title, vector<cv::Point3d> &pts);
	std::vector<cv::Point3d> get_face_region(cv::Mat && faceImg, std::string && title);


private:
	const string* winName;
	mutable const Mat* image;
	Mat mask;
	Mat leftEyeModel, rightEyeModel, faceModel;

	uchar rectState;
	bool isInitialized;

	Rect rect;
	vector<Point> facePxls, leftEyePxls, rightEyePxls;
	int iterCount;

	Scalar RED, BLUE, GREEN;
	int FACE_KEY, LEFT_KEY, RIGHT_KEY;

    double draw_height;
    bool save_img;
    vector<Point2i> draw_pts2d;
    std::string draw_title;
    vector<cv::Point3d> draw_pts3d;
};
