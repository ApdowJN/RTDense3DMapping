//
// Created by wangweihan on 5/2/22.
//

#ifndef IROS_STEREOMODULE_H
#define IROS_STEREOMODULE_H
#include <opencv2/opencv.hpp>
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

using namespace std;
using namespace std::chrono;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef unsigned int uint;
typedef unsigned long long ull;

// public functions
void runStereo(int ndisp, int wsize, bool post, string method, const string &filename, const cv::Mat &imLeft, const cv::Mat &imRight, cv::Mat &disp, cv::Mat &conf, cv::Mat &cost1, cv::Mat &cost2, cv::Mat &cost3);

void sad(uint8* leftp, uint8 *rightp, uint *cost, int* shape, int ndisp, int wsize);
void ncc(uint8* leftp, uint8 *rightp, double* cost, int* shape, int ndisp, int wsize);

void colorMat2Array(const cv::Mat &imColor);
void grayMat2Array(const cv::Mat &imGray, uint8_t* pGray);

#endif //IROS_STEREOMODULE_H
