//
// Created by wangweihan on 9/6/22.
//

#ifndef IROS_POINTCLOUDBUILDER_H
#define IROS_POINTCLOUDBUILDER_H
#include <list>
#include <thread>
#include <mutex>
#include "util.h"
#include "Frame.h"
class PointCloudBuilder {
public:
    PointCloudBuilder(const int &n_fused_frame);
    void InsertPointCloud(const cv::Mat &fusedMap, const Frame &referenceFrame, string pcl_path);
    void GeneratePointCloud();
    bool CheckNewPointCloud();
    void Run();
    bool isStop();
    int mNTotalFusedFrame;
    int mNCurrentFusedFrame;
    double mTotal_pcl_time;
    PointCloud::Ptr globalMap;
    std::list<cv::Mat> mlFusedMap;
    std::list<Frame> mlFrame;
    std::list<string> mlpcl_path;
    std::mutex mMutexPointCloud;
};
#endif //IROS_POINTCLOUDBUILDER_H
