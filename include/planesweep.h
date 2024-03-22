//
// Created by wangweihan on 12/3/22.
//

#ifndef IROS_PLANESWEEP_H
#define IROS_PLANESWEEP_H
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <sstream>
#include <chrono>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "Frame.h"

using namespace std;
using namespace std::chrono;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef unsigned int uint;
typedef unsigned long long ull;

class PlaneSweep{
public:
    PlaneSweep();

    PlaneSweep(const int &width, const int &height);

    void configPlanes(float min_depth, float max_depth, int num_planes);

    double* Homography(const cv::Mat &K_ref_inv, const cv::Mat &Rcr, const cv::Mat &tcr, const cv::Mat &K_ref, std::vector<double> normal, double d);

    uint8 get_val(const cv::Mat &img, double x, double y);

    // pixels are out of bounds are set to zero.
    void interpolate(const int &globalId, double* H,uint8 *im_h );

    void SAD( uint8* im_h, int w,int d);

    void WTA();

    void sweep(const deque<Frame> &keyframesCache, std::vector<double> normal);

    ~PlaneSweep();

    std::vector<cv::Mat> imgs;
    cv::Mat depth_map;

private:
    std::vector<double> plane_depths;
    double depth_interval;
    double* volume;

    int ref_global_id;

    int mwidth;
    int mheight;
};
#endif //IROS_PLANESWEEP_H
