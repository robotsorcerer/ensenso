#include <iostream>
#include <typeinfo>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#//include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

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
    if( draw_img.empty() || winName->empty() )
        return;

    draw_img = _image;
    winName = &_winName;

    this->reset();
}

void CoordinatesPicker::showImage() const
{
    if( draw_img.empty() || winName->empty() )
        return;

    Mat res, mask;

    if( !isInitialized )
        draw_img.copyTo( res );
    else
        draw_img.copyTo( mask );

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
        {
            // waitKey(1);
            // wm_->destroyAllWindows();
        }
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
            point3d = p3d[0];
        }
//        sprintf(text, "(%3.3f, %3.3f, %3.3f)", point3d.x, point3d.y, point3d.z);
        cv::putText(img3, text, cv::Point(x,y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);

        draw_pts3d.push_back(point3d);
        draw_pts2d.push_back(cv::Point2i(x,y));
        int lineThickness = 2;
        if(draw_pts2d.size() > 1)
            cv::line(img3, draw_pts2d[draw_pts2d.size()-2], draw_pts2d[draw_pts2d.size()-1],cv::Scalar(255,0,0), lineThickness, cv::LINE_AA);
        if(draw_pts2d.size() == 4)
            cv::line(img3, draw_pts2d[0], draw_pts2d[draw_pts2d.size()-1],cv::Scalar(255,0,0), lineThickness, cv::LINE_AA);
        winName = &draw_title;
        img3.copyTo(draw_img);
        // showImage(draw_title, img3);
        showImage();
        // wm_->imshow(std::move(draw_title), std::move(img3));
    }
}

int CoordinatesPicker::getCount() const {return iterCount; }


static void onMouse(int event, int x, int y, int flags, void* param)
{
    CoordinatesPicker* ptr = reinterpret_cast<CoordinatesPicker*>(param);
    ptr->mouseClick( event, x, y, flags, 0 );
}

void CoordinatesPicker::get_points(cv::Mat&& img, std::string&& title, vector<cv::Point3d> &pts)
{
    draw_title = title;

    // wm_ = boost::shared_ptr<WindowManager>(new WindowManager(title));
    setMouseCallback(title, &onMouse, 0 );

    if(img.type() != 0 && img.type() != 8 && img.type() != 16 && img.type() != 24)
    {
        cv::Mat img2;
        double minVal, maxVal;
        cv::minMaxIdx(img, &minVal, &maxVal);
        img.convertTo(img2, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
        cv::equalizeHist(img2, draw_img);
        cv::cvtColor(draw_img, draw_img, CV_GRAY2RGB);
    }
    else
    {
        // ROS_INFO("in get_points and copying to draw_img");
        img.copyTo(draw_img);
        // ROS_INFO("in get_points and finished copying to draw_img");
    }
    // wm_->imshow(std::move(title), std::move(draw_img));
    namedWindow(title, WINDOW_NORMAL);
    cv::imshow(title, img);

    int key = waitKey(1);

    switch(key & 0xFF )
    {
      case 'q':
        destroyAllWindows();
        cv::waitKey(100);
        break;
      default:
        break;
    }

    // std::cout << "draw_img size: " << draw_img.size() << std::endl;
    winName = &title;
    // showImage();
    pts = draw_pts3d;
    draw_pts3d.clear();
    draw_pts2d.clear();
}

std::vector<cv::Point3d> CoordinatesPicker::get_face_region(cv::Mat && faceImg, std::string && title)
{
    std::vector<cv::Point3d> pts;
    get_points(std::move(faceImg), std::move(title), pts);

    return pts;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_loader_node");

    std::unique_ptr<ImagesLoader> il (new ImagesLoader());
    std::unique_ptr<CoordinatesPicker> cl (new CoordinatesPicker());

    // il->get_files();
    std::vector<std::string> files_vector = il->filename_full;

    std::string what_to_do;
    cv::Mat faceImg; //container for storing faces

    //load all files one by one
    // for(auto it=files_vector.cbegin(); it != files_vector.cend(); ++it)
    boost::shared_ptr<WindowManager> wm_ = boost::shared_ptr<WindowManager>(new WindowManager());
    auto it = files_vector.cbegin();
    while( it != files_vector.cend() && ros::ok() )
    {
        ROS_INFO("Please draw the bounding boxes for face, and eyes in the shown image");
        ROS_INFO("Type {n} or {next} on the cmdline to advance to the next image");

        cl->draw_img = imread(*it, IMREAD_ANYDEPTH );

        // for(auto i = 0; i < cl->draw_img.rows; ++i)
        // {
        //   for(auto j = 0; j < cl->draw_img.cols; ++j)
        //     OUT("pixel at img [" << i << ", " << j << "]" << cl->draw_img.at(i, j) );
        // }

        std::string file_name = *it;
        // std::cout << "faceImg Size: " << cl->draw_img.size() << "\t" << file_name << std::endl;

        cl -> get_face_region(std::move(cl->draw_img), std::move(file_name));

        std::getline(std::cin, what_to_do);
        if( !what_to_do.compare("n") || !what_to_do.compare("next" ) )
        {
            wm_->clearWindow(std::move(file_name), std::move(cl->draw_img));
            ++it;
        }
        else if( !what_to_do.compare("q") || !what_to_do.compare("quit") )
        {
            ROS_INFO("Ground Truthing of bounding boxes finished");
            goto exit_main;
            break;
        }
        else
        {
            ROS_INFO("Wrong item entered");
        }
    }

    // ros::spin();
  exit_main:
    destroyAllWindows();
    ros::shutdown();
    return EXIT_SUCCESS;
}
