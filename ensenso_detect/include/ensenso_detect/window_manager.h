/*
*  Description: Utility for displaying opencv windows
*/

#ifndef WINDOW_MANAGER_H_
#define WINDOW_MANAGER_H_

//#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <string>

class WindowManager
{
public:
   WindowManager();
   WindowManager(const std::string mainWinTitle);
   // Use this to let the WindowManager know what the screen resolution is so that
   // it can properly size the images.  Assumed to be 1920 x 1080 by default
   void setScreenResolution(const int width, const int height);
   // Adds an image to the window manager and displays all current images
   void imshow(const std::string winTitle, const cv::Mat img);
   // Adds a float image to the window manager and displays all current images
   void imshowFloat(const std::string winTitle, const cv::Mat img);


   // Add a window to the WindowManager (doesn't display anything though)
   // if the window title already exists, the corresponding image is overwritten
   void addWindow(const std::string winTitle, const cv::Mat img);
   // Add a window to the WindowManager and convert to integer values
   // if the window title already exists, the corresponding image is overwritten
   void addWindowFloat(const std::string winTitle, const cv::Mat img);
//    // Shows one window using full specified resolution in the center of the screen
//    bool showWindowFullScreen(const std::string winTitle) const;
   // Shows all the windows tiled - note that toolbars which occupy some of the
   // screen tend to make the window tile imperfectly.  Pauses with waitKey.
   void showAllWindowsTiled();
   void showAllWindowsTiledSingle();
   void clearAllWindows();
   // Removes all window from the display
   void destroyAllWindows();

private:
   void showAllWindowsTiled(const bool useSingleWindow, const std::string winTitle);

   int screenResWidth_;
   int screenResHeight_;
   std::string mainWindowTitle_;
   std::vector<std::string> windowTitles_;
   std::vector<cv::Mat> imgs_;
};

void imshowFloat(const std::string winTitle, const cv::Mat img);

#endif // WINDOW_MANAGER_H_
