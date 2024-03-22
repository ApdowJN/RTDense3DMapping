#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/sfm/fundamental.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <dirent.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <pcl/visualization/cloud_viewer.h>

#include "util.h"
#include "Frame.h"
#include "fusion.h"

using namespace std;
using namespace cv;

void confidence_fusion(const cv::Size size, cv::Mat &fused_map, cv::Mat &fused_conf, const cv::Mat &K, const vector<Mat> &P, const deque<Frame> &views, const long unsigned int &sequenceId, const float &conf_pre_filt, const float &conf_post_filt, const float &support_ratio)
{
    int num_views = views.size(); //global camera id
    cout << "Rendering depth maps (ref #"<<sequenceId <<")....";

    vector<Mat> depth_refs;
    vector<Mat> conf_refs;
    const int rows = size.height;
    const int cols = size.width;
    int mid = num_views/2;

    const float fx = K.at<float>(0,0);
    const float invfx = 1.0f/fx;
    const float fy = K.at<float>(1,1);
    const float invfy = 1.0f/fy;
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    unordered_map<int, Frame> mapLocalId2Frame;
    Frame ReferenceFrame = views.at(mid);
    // for each supporting view of the current index (reference view)
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < (int) views.size(); i++)
    {
        Frame frame = views.at(i);
        mapLocalId2Frame[i] = frame;
        if(frame.mIndex == sequenceId)
        {
            // push the current view
            depth_refs.push_back(frame.mDepthMap);
            conf_refs.push_back(frame.mConfMap);
            continue;
        }
        Mat depth_ref = Mat::zeros(size, CV_32F);
        Mat conf_ref = Mat::zeros(size, CV_32F);


#pragma omp parallel num_threads(12)
        {
#pragma omp for collapse(2)
            for (int v=0; v<rows; ++v)
            {
                for (int u=0; u<cols; ++u)
                {
                    const float depth = frame.mDepthMap.at<float>(v,u);
                    const float conf = frame.mConfMap.at<float>(v,u);

                    if(conf < conf_pre_filt)
                    {
                        continue;
                    }

                    if(depth>0)
                    {
                        // compute 3D world coord of back projection
                        const float x = (u-cx)*depth*invfx;
                        const float y = (v-cy)*depth*invfy;
                        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, depth);
                        Mat Tcw = frame.mTcw;

                        Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
                        Mat Rwc = Rcw.t();
                        Mat tcw = Tcw.rowRange(0,3).col(3);
                        Mat twc = -Rcw.t()*tcw;

                        //get mappoint in world frame
                        Mat x3Dw =  Rwc*x3Dc+twc;

                        // calculate pixel location in reference image
                        // Transform into Reference Camera Coords.
                        Mat Trw = ReferenceFrame.mTcw; //pose Tref_world

                        Mat Rrw = Trw.rowRange(0,3).colRange(0,3);
                        Mat trw = Trw.rowRange(0,3).col(3);

                        cv::Mat x3Dr = Rrw*x3Dw+trw;

                        const float xr = x3Dr.at<float>(0);
                        const float yr = x3Dr.at<float>(1);
                        const float invzr = 1.0/x3Dr.at<float>(2);

                        if(invzr<0)
                            continue;

                        int u_ref = (int)floor(fx*xr*invzr+cx);
                        int v_ref = (int)floor(fy*yr*invzr+cy);

                        if (u_ref < 0 || u_ref >= size.width || v_ref < 0 || v_ref >= size.height)
                        {
                            continue;
                        }

                        float proj_depth = x3Dr.at<float>(2);

                        /*
                         * Keep the closer (smaller) projection depth.
                         * A previous projection could have already populated the current pixel.
                         * If it is 0, no previous projection to this pixel was seen.
                         * Otherwise, we need to overwrite only if the current estimate is closer (smaller value).
                         */
                        if (depth_ref.at<float>(v_ref,u_ref) > 0)
                        {
                            if(depth_ref.at<float>(v_ref,u_ref) > proj_depth)
                            {
                                depth_ref.at<float>(v_ref,u_ref) = proj_depth;
                                conf_ref.at<float>(v_ref,u_ref) = conf;
                            }
                        }
                        else
                        {
                            depth_ref.at<float>(v_ref,u_ref) = proj_depth;
                            conf_ref.at<float>(v_ref,u_ref) = conf;
                        }

                    }
                }
            }
} //omp parallel
        depth_refs.push_back(depth_ref);
        conf_refs.push_back(conf_ref);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/1000;
    cout << "   "<<(float)duration <<"ms"<<  endl;

    cout << "Fusing depth maps...";
    begin = std::chrono::high_resolution_clock::now();
    for (int v=0; v<rows; ++v) {
        for (int u=0; u<cols; ++u) {
            float f = 0.0;
            float initial_f = 0;
            float C = 0.0;
            int initial_lid = 0;

            //take most confident pixel as initial depth estimate
            for (int lid = 0; lid < num_views; ++lid) {
                if (conf_refs[lid].at<float>(v,u) > C) {
                    f = depth_refs[lid].at<float>(v,u);
                    C = conf_refs[lid].at<float>(v,u);
                    initial_f = f;
                    initial_lid = lid;
                }
            }

            // Set support region as fraction of initial depth estimate
            float epsilon = support_ratio * initial_f;

            for (int lid = 0; lid < num_views; ++lid) {
                if(lid == initial_lid)
                    continue;
                // grab current depth and confidence values

                float curr_depth = depth_refs[lid].at<float>(v,u); //0~5
                float curr_conf = conf_refs[lid].at<float>(v,u); //there is nan here

                // if depth is within the support region of the initial depth
                if (abs(curr_depth - initial_f) <= epsilon) {
                    if((C + curr_conf) != 0) {
                        float f1 = ((f*C) + (curr_depth*curr_conf)) / (C + curr_conf);
                        if(isnan(f1))
                            cerr<<"error value in confidence map"<<endl;
                        f = ((f*C) + (curr_depth*curr_conf)) / (C + curr_conf);

                    }
                    C += curr_conf;
                }
                    // if depth is closer than initial estimate (occlusion)
                else if(curr_depth < initial_f) {
                    C -= curr_conf;
                }
                    // if depth is farther than initial estimate (free-space violation)
                else if(curr_depth > initial_f) {

                    const float xr = (u-cx)*initial_f*invfx;
                    const float yr = (v-cy)*initial_f*invfy;

                    cv::Mat x3Dr = (cv::Mat_<float>(3,1) << xr, yr, initial_f);

                    Mat Trw = mapLocalId2Frame[initial_lid].mTcw;
                    Mat Rrw = Trw.rowRange(0,3).colRange(0,3);
                    Mat Rwr = Rrw.t();
                    Mat trw = Trw.rowRange(0,3).col(3);
                    Mat twr = -Rrw.t()*trw;
                    Mat x3Dw = Rwr*x3Dr+twr;

                    Mat Tcw = mapLocalId2Frame[lid].mTcw;
                    Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
                    Mat tcw = Tcw.rowRange(0,3).col(3);

                    cv::Mat x3Dc = Rcw*x3Dw+tcw;
                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0/x3Dc.at<float>(2);

                    if(invzc<0)
                        continue;

                    int u_c = (int)floor(fx*xc*invzc+cx);
                    int v_c = (int)floor(fy*yc*invzc+cy);

                    if (u_c >= 0 && u_c < size.width && v_c >= 0 && v_c < size.height) {
                        C -= conf_refs[lid].at<float>(v_c,u_c);
                    }

                }

            }

            // drop any estimates that do not meet the minimum confidence value
            if (C <= conf_post_filt) {
                f = -1.0;
                C = -1.0;
            }

            // set the values for the confidence and depth estimates at the current pixel
            fused_map.at<float>(v,u) = f;
            fused_conf.at<float>(v,u) = C;

        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/1000;
    cout << "   "<<(float)duration <<"ms"<<  endl;

    cout << "Smoothing / Hole Filling...";
    // hole-filling parameters
    begin = std::chrono::high_resolution_clock::now();
    int w = 3;
    int w_offset = (w-1)/2;
    int w_inliers = (w*w)/4;

    // smoothing parameters
    int w_s = 3;
    int w_s_offset = (w_s-1)/2;
    int w_s_inliers = (w_s*w_s)/4;

    Mat filled_map = Mat::zeros(size, CV_32F);
    Mat smoothed_map = Mat::zeros(size, CV_32F);

    // Fill in holes (-1 values) in depth map
    for (int r=w_offset; r<rows-w_offset; ++r) {
        for (int c=w_offset; c<cols-w_offset; ++c) {
            if (fused_map.at<float>(r,c) < 0.0){
                filled_map.at<float>(r,c) = med_filt(fused_map(Rect(c-w_offset,r-w_offset,w,w)), w, w_inliers);
            } else {
                filled_map.at<float>(r,c) = fused_map.at<float>(r,c);
            }
        }
    }

    // Smooth out inliers
    for (int r=w_s_offset; r<rows-w_s_offset; ++r) {
        for (int c=w_s_offset; c<cols-w_s_offset; ++c) {
            if (filled_map.at<float>(r,c) != -1){
                smoothed_map.at<float>(r,c) = med_filt(filled_map(Rect(c-w_s_offset,r-w_s_offset,w_s,w_s)), w_s, w_s_inliers);
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/1000;
    cout << "   "<<(float)duration <<"ms"<<  endl;

    fused_map = smoothed_map;
}
