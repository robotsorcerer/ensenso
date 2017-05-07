#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <ensenso_detect/bboxes.h>

//class constructor impl
ImagesLoader::ImagesLoader()
:verbose(false)
{
    if(get_files()) //populate faces_image paths
        ROS_INFO("populated vectors of filenames");
}

//class destructor impl
ImagesLoader::~ImagesLoader()
{

}

bool ImagesLoader::get_images_path(boost::filesystem::path && facePosPath)
{
    if(!pathfinder::getROSPackagePath("ensenso_detect", detectPath))
    {
        ROS_INFO("given ensenso_detect path does not exist");
        return false;
    }

    facePosPath = detectPath / "manikin" / "raw" / "face_images";

    return true;
}

bool ImagesLoader::get_files()
{
    boost::filesystem::path facePosPath;

    if(get_images_path(std::move(facePosPath)))
        ROS_INFO("\nFound face images path at %s ", (facePosPath.c_str()));

    for(boost::filesystem::directory_iterator end, iter(facePosPath.c_str()); iter!= end; ++iter)
    {
        boost::filesystem::path fn = (*iter).path().filename();
        boost::filesystem::path srcFile = (*iter).path();
        if(verbose)
            ROS_INFO("reading file: %s ", fn.c_str());
        // ROS_INFO("full path of file src: %s", srcFile.c_str());

        filename_abs.push_back(fn.c_str());
        filename_full.push_back(srcFile.c_str());
    }

    return true;
}

/*Implement Coordinates getters*/
CoordinatesPicker::CoordinatesPicker()
    : RED(Scalar(0, 0, 255)), BLUE(Scalar(255, 0, 0)),
    GREEN(Scalar(0, 255, 0) ), FACE_KEY(EVENT_FLAG_CTRLKEY),
    LEFT_KEY(EVENT_FLAG_SHIFTKEY), RIGHT_KEY(EVENT_FLAG_ALTKEY)
{

};

//destructor
CoordinatesPicker::~CoordinatesPicker() {}

void CoordinatesPicker::reset()
{
    facePxls.clear();
    leftEyePxls.clear();
    rightEyePxls.clear();

    isInitialized = false;
    rectState = NOT_READY;
    iterCount = 0;
}

void CoordinatesPicker::setImageAndWinName( Mat&& _image, string && _winName )
{
    if( image->empty() || winName->empty() )
        return;

    image = &_image;
    winName = &_winName;

    this->reset();
}

void CoordinatesPicker::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res, mask;

    if( !isInitialized )
        image->copyTo( res );
    else
        image->copyTo( mask );

    // vector<Point>::const_iterator it;
    for(auto cit = rightEyePxls.cbegin(); cit != rightEyePxls.cend(); ++cit )
        circle( res, *cit, radius, BLUE, thickness );

    for(auto cit = leftEyePxls.cbegin(); cit != leftEyePxls.cend(); ++cit)
        circle( res, *cit, radius, BLUE, thickness );

    if(rectState == IN_VOGUE || rectState == READY )
        rectangle(res, Point( rect.x, rect.y ), Point( rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    imshow( *winName, res );
}



void CoordinatesPicker::nextIter()
{

}

void CoordinatesPicker::mouseClick(int event, int x, int y, int flags, void* )
{
    if (event == cv::EVENT_MOUSEMOVE)    {
        if(draw_pts3d.size() == 4)
            wm_->destroyAllWindows();
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN)    {
    }
    else if(event == cv::EVENT_LBUTTONUP)        {
        char text[256];
        cv::Mat img3=draw_img;

        //first, lets get each point in the calib frame
        cv::Point3d point3d;
        if(save_img)
        {
            point3d.x = x;
            point3d.y = y;
            point3d.z = 0;
        }
        else
        {
            vector<cv::Point2i> p2d(1, cv::Point2i(x,y));
            vector<cv::Point3d> p3d(1);
            // this->intersectZPlane(draw_prm, draw_height, p2d, p3d);
            point3d = p3d[0];
        }
//        sprintf(text, "(%3.3f, %3.3f, %3.3f)", point3d.x, point3d.y, point3d.z);
        cv::putText(img3, text, cv::Point(x,y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);

        draw_pts3d.push_back(point3d);
        draw_pts2d.push_back(cv::Point2i(x,y));
        int lineThickness = 2;
        if(draw_pts2d.size() > 1)
            cv::line(img3, draw_pts2d[draw_pts2d.size()-2], draw_pts2d[draw_pts2d.size()-1],cv::Scalar(255,0,0), lineThickness, cv::LINE_AA);
        if(draw_pts2d.size() == 4) // Close the polygon if we have 4 points
            cv::line(img3, draw_pts2d[0], draw_pts2d[draw_pts2d.size()-1],cv::Scalar(255,0,0), lineThickness, cv::LINE_AA);
        wm_->imshow(draw_title, img3);
    }
}

int CoordinatesPicker::getCount() const {return iterCount; }


void onMouse(int event, int x, int y, int flags, void* param)
{
    CoordinatesPicker* ptr = reinterpret_cast<CoordinatesPicker*>(param);
    ptr->mouseClick( event, x, y, flags, 0 );
}

void CoordinatesPicker::get_points(cv::Mat&& img, const std::string& title, vector<cv::Point3d> &pts)
{
    draw_title = title;
    wm_ = boost::shared_ptr<WindowManager>(new WindowManager(title));
    setMouseCallback(title, &onMouse, this);
    wm_->imshow(title, draw_img);
    wm_->destroyAllWindows();
    pts = draw_pts3d;
    draw_pts3d.clear();
    draw_pts2d.clear();
}

std::vector<cv::Point3d> CoordinatesPicker::get_face_region(cv::Mat && faceImg, const std::string & title)
{
    std::vector<cv::Point3d> pts;
    get_points(std::move(faceImg), title, pts);

    return pts;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_loader_node");

    std::unique_ptr<ImagesLoader> il (new ImagesLoader());
    std::unique_ptr<CoordinatesPicker> cl (new CoordinatesPicker());

    // il->get_files();
    std::vector<std::string> files_vector = il->filename_full;

    for(auto it=files_vector.cbegin(); it != files_vector.cend(); ++it)
    {
        cv::Mat faceImg = imread(*it, IMREAD_ANYDEPTH );
        cl -> get_face_region(std::move(faceImg), *it);
    }


    ros::spin();

    return EXIT_SUCCESS;
}
