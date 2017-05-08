#include <ensenso_detect/window_manager.h>
#include <limits>
#include <string>

using namespace cv;

void imshowFloat(const std::string winTitle, const cv::Mat img)
{
    if (!img.empty())
    {
        cv::Mat imgConv;
        double minVal, maxVal;
        cv::minMaxIdx(img, &minVal, &maxVal);
        img.convertTo(imgConv, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::imshow(winTitle, imgConv);
    }
}

WindowManager::WindowManager() :
        screenResWidth_(1280), screenResHeight_(1024), mainWindowTitle_("")
{
    // cv::namedWindow(mainWindowTitle_, WINDOW_NORMAL);
    startWindowThread();
}

WindowManager::WindowManager(const std::string winTitle) :
        screenResWidth_(1280), screenResHeight_(1024)
{
    mainWindowTitle_ = winTitle;
    // cv::namedWindow(mainWindowTitle_, WINDOW_NORMAL);
    startWindowThread();
}

void WindowManager::imshow(std::string&& winTitle, cv::Mat&& img)
{
    int key = waitKey(1);
    if (!img.empty())
    {
          namedWindow(winTitle, WINDOW_NORMAL);
          resizeWindow(winTitle, img.rows, img.cols);
          cv::imshow(winTitle, img);

          OUT("winTitle: " << winTitle )

          switch(key & 0xFF )
          {
            case 'q':
              destroyAllWindows();
              break;
            default:
              break;
          }
    }
}

void WindowManager::clearWindow(std::string&& winTitle, cv::Mat&& img)
{
    winTitle.clear();
    img.release();
}

void WindowManager::setScreenResolution(const int width, const int height)
{
    screenResWidth_ = width;
    screenResHeight_ = height;
}
