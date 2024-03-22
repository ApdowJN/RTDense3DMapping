//
// Created by wangweihan on 10/17/21.
//

#ifndef ICRA_FRAME_H
#define ICRA_FRAME_H
#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

class Frame {
public:
    // public parameters
	cv::Mat mK;
    cv::Size mSize;
    float mBaseline;
    long unsigned int mIndex;
//    cv::Mat mRcw; // Rotation from world to camera
//    cv::Mat mtcw; // Translation from world to camera
    cv::Mat mTcw;
    cv::Mat mDepthMap;
    cv::Mat mConfMap;
    cv::Mat mColorMap;
    std::vector<float> mvuRight;

    // Constructor for stereo cameras.
    Frame();
    Frame(const Frame &frame);
    Frame(cv::Mat &K, const float &baseline, const long unsigned int &sequenceId, const cv::Size &imgSize);

	// public functions
    void setPose(cv::Mat Tcw_);
    void setColorMap(cv::Mat imColor_);
    void getDepthMap();
    void getDepthMap(const cv::Mat &imDisparity);

    void load_confidence_map(const cv::Size &imageSize, const string &filePath);
    void load_depth_map(const cv::Size &imageSize, const cv::Mat &dispMat, const float &baseline);

    void parabola_fitting(const cv::Mat &cost1, const cv::Mat &cost2, const cv::Mat &cost3, const cv::Mat &imDisparity);

};
#endif //ICRA_FRAME_H
