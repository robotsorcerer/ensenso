/*
*  Description: Utility for displaying opencv windows
*/

#ifndef WINDOW_MANAGER_H_
#define WINDOW_MANAGER_H_

//#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <string>

#define OUT(__o__) std::cout << __o__ << std::endl;

class WindowManager
{
public:
   WindowManager();
   WindowManager(const std::string mainWinTitle);
   // Use this to let the WindowManager know what the screen resolution is so that
   // it can properly size the images.  Assumed to be 1920 x 1080 by default
   void setScreenResolution(const int width, const int height);
   // Adds an image to the window manager and displays all current images
   void imshow(std::string&& winTitle, cv::Mat&& img);
   //myne for showing win contents
  //  void showWindow(const std::string winTitle, const cv::Mat img);
   //clear window namespace
   void clearWindow(std::string&& winTitle, cv::Mat&& img);

private:

   int screenResWidth_;
   int screenResHeight_;
   std::string mainWindowTitle_;
   std::vector<std::string> windowTitles_;
   std::vector<cv::Mat> imgs_;
};

#endif // WINDOW_MANAGER_H_
