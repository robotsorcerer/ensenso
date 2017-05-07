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
        screenResWidth_(1920), screenResHeight_(1080), mainWindowTitle_("")
{
    cv::namedWindow(mainWindowTitle_, WINDOW_NORMAL);
    startWindowThread();
}

WindowManager::WindowManager(const std::string winTitle) :
        screenResWidth_(1920), screenResHeight_(1080)
{
    mainWindowTitle_ = winTitle;
    cv::namedWindow(mainWindowTitle_, WINDOW_NORMAL);
    startWindowThread();
}

void WindowManager::imshow(const std::string winTitle, const cv::Mat img)
{
    if (!img.empty())
    {
        addWindow(winTitle, img);
        showAllWindowsTiledSingle();
    }
}

void WindowManager::imshowFloat(const std::string winTitle, const cv::Mat img)
{
    if (!img.empty())
    {
        cv::Mat imgConv;
        double minVal, maxVal;
        cv::minMaxIdx(img, &minVal, &maxVal);
        std::cout << "Min and Max: " << minVal << "     " << maxVal << std::endl;
        img.convertTo(imgConv, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::imshow("imgConv", imgConv);
//	    addWindow(winTitle, imgConv);
//		showAllWindowsTiledSingle();
    }
}

void WindowManager::setScreenResolution(const int width, const int height)
{
    screenResWidth_ = width;
    screenResHeight_ = height;
}

void WindowManager::addWindow(const std::string winTitle, const cv::Mat img)
{
    bool winFound = false;

    for (int ii = 0; ii < windowTitles_.size(); ii++)
    {
        if (!windowTitles_[ii].compare(winTitle))
        {
            winFound = true;
            imgs_[ii] = img;
        }
    }
    if (!winFound)
    {
        windowTitles_.push_back(winTitle);
        imgs_.push_back(img);
    }
}

void WindowManager::addWindowFloat(const std::string winTitle, const cv::Mat img)
{
    cv::Mat imgConv;
    double minVal, maxVal;
    cv::minMaxIdx(img, &minVal, &maxVal);
    img.convertTo(imgConv, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    addWindow(winTitle, imgConv);
}

void WindowManager::showAllWindowsTiled()
{
    showAllWindowsTiled(false, "");
}
void WindowManager::showAllWindowsTiledSingle()
{
    showAllWindowsTiled(true, mainWindowTitle_);
}

void WindowManager::showAllWindowsTiled(const bool useSingleWindow, const std::string winTitle)
{
    int imgHeight = imgs_[0].rows;
    int imgWidth = imgs_[0].cols;
    int numImgs = imgs_.size();

    // Find the best number of rows and columns to use when tiling the images
    // Currently assumes all images are the same size
    int bestCols, bestRows;
    double bestDiff = std::numeric_limits<double>::max();

    double desiredRatio = (double)screenResWidth_ / (double)screenResHeight_;
    for (int cols = 1; cols <= numImgs; cols++)
    {
        int rows = ceil((double)numImgs / (double)cols);
        double currentRatio = (double)(cols * imgWidth) / (double)(rows * imgHeight);
        double diff = fabs(currentRatio - desiredRatio);
        if (diff < bestDiff)
        {
            bestCols = cols;
            bestRows = rows;
            bestDiff = diff;
        }
    }

//    std::cout << "Best cols: " << bestCols << "   Best rows: " << bestRows << std::endl;

    double scaleFactorHeight = (double)screenResHeight_ / (double)(imgHeight * bestRows);
    double scaleFactorWidth = (double)screenResWidth_ / (double)(imgWidth * bestCols);
    double scaleFactor = min(scaleFactorHeight, scaleFactorWidth);
    double scaledImageHeight = scaleFactor * imgHeight;
    double scaledImageWidth = scaleFactor * imgWidth;

    if (useSingleWindow)
    {
        int bigImageRows = bestRows * imgHeight;
        int bigImageCols = bestCols * imgWidth;
        Mat bigImage(bigImageRows, bigImageCols, CV_8UC3);

        for (int ii = 0; ii < numImgs; ii++)
        {
            int col = ii % bestCols + 1;
            int row = ii / bestCols + 1;
            Mat roi = bigImage(Rect(col * imgWidth - imgWidth, row * imgHeight - imgHeight, imgWidth, imgHeight));
            imgs_[ii].copyTo(roi);
        }
        cv::imshow(winTitle, bigImage);
        double scaleFactor = 1.0 * ((double)screenResHeight_ / (double)imgHeight);
        resizeWindow(winTitle, scaleFactor * imgWidth, scaleFactor * imgHeight);
        moveWindow(winTitle, 0, 0);
        waitKey(0);
//        waitKey(1);
//        destroyWindow(winTitle);
//        waitKey(1);

    }
    else
    {
        for (int ii = 0; ii < numImgs; ii++)
        {
            int col = ii % bestCols + 1;
            int row = ii / bestCols + 1;

            cv::namedWindow(windowTitles_[ii], WINDOW_NORMAL);
            startWindowThread();
            cv::imshow(windowTitles_[ii], imgs_[ii]);
            resizeWindow(windowTitles_[ii], scaledImageWidth, scaledImageHeight);
            int imgPositionX = col * scaledImageWidth - scaledImageWidth;
            int imgPositionY = row * scaledImageHeight - scaledImageHeight;
            //std::cout << "Image X: " << imgPositionX << "   Image Y: " << imgPositionY << std::endl;
            moveWindow(windowTitles_[ii], imgPositionX, imgPositionY);
        }
        waitKey(0);
    }
}

void WindowManager::clearAllWindows()
{
    windowTitles_.clear();
    imgs_.clear();
}

void WindowManager::destroyAllWindows()
{
    waitKey(1);
    destroyWindow(mainWindowTitle_);
    waitKey(1);
}
//
// int main(int argc, char** argv)
// {
//     Mat img1 =
//             imread("/home/local/armsdev/arms_ws/src/ar_product_data/regression_test/big_wooden_cube_on_origin/frames/14425708_image_0.png");
//     Mat img2 =
//             imread("/home/local/armsdev/arms_ws/src/ar_product_data/regression_test/big_wooden_cube_on_origin/frames/14425724_image_0.png");
//
// #if 0
//     WindowManager wm1("Basic Window");
//     wm1.imshow("Tmp1", img1);
//     wm1.imshow("Tmp2", img2);
//     wm1.imshow("Tmp3", img1);
//     wm1.destroyAllWindows();
// #endif
//
// #if 1
//     WindowManager wm2("Add Window Test");
//     wm2.addWindow("Tmp1", img1);
//     wm2.addWindow("Tmp2", img2);
//     wm2.showAllWindowsTiledSingle();
//     wm2.destroyAllWindows();
// #endif
//
// #if 0
//     WindowManager wm3("Multiple tiled windows");
//     wm3.addWindow("Tmp1", img1);
//     wm3.addWindow("Tmp2", img2);
//     wm3.showAllWindowsTiled();
//     wm3.destroyAllWindows();
// #endif
//
// }
