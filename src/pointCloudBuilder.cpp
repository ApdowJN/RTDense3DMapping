//
// Created by wangweihan on 9/6/22.
//
#include "pointCloudBuilder.h"
#include "util.h"
PointCloudBuilder::PointCloudBuilder(const int &n_fused_frame): mNTotalFusedFrame(n_fused_frame),
mNCurrentFusedFrame(0),mTotal_pcl_time(0.0), globalMap(new PointCloud) {}

void PointCloudBuilder::InsertPointCloud(const cv::Mat &fusedMap, const Frame &referenceFrame, string pcl_path) {
    std::unique_lock<std::mutex> lock(mMutexPointCloud);
    mlFusedMap.push_back(fusedMap);
    mlFrame.push_back(referenceFrame);
    mlpcl_path.push_back(pcl_path);
}

bool PointCloudBuilder::isStop() {
    if(mNCurrentFusedFrame >= mNTotalFusedFrame)
        return true;
    else
        return false;
}

bool PointCloudBuilder::CheckNewPointCloud() {
    std::unique_lock<std::mutex> lock(mMutexPointCloud);
    return (!mlFrame.empty());
}

void PointCloudBuilder::GeneratePointCloud() {
    Frame ReferenceFrame;
    cv::Mat fused_map;
    string pcl_path;
    {
        std::unique_lock<std::mutex> lock(mMutexPointCloud);
        ReferenceFrame = mlFrame.front();
        fused_map= mlFusedMap.front();
        pcl_path = mlpcl_path.front();
    }


    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    PointCloud::Ptr cloud = generated_point_clouds(ReferenceFrame.mColorMap, fused_map, ReferenceFrame.mTcw, ReferenceFrame.mK);
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
    mTotal_pcl_time += duration;
    cout << "Duration of point cloud generation: " << duration << "ms" << endl;

    (*globalMap) += *cloud; // Add current 3D model into whole model

    int l = pcl_path.length();
    pcl_path = pcl_path.substr(0,l-3) + "ply";
    // pcl::io::savePLYFileASCII(pcl_path, *cloud);
    pcl::io::savePLYFileBinary(pcl_path, *cloud);
    cloud->clear();

    {
        std::unique_lock<std::mutex> lock(mMutexPointCloud);
        mlFusedMap.pop_front();
        mlFrame.pop_front();
        mlpcl_path.pop_front();
    }


}

void PointCloudBuilder::Run() {
    // Generate 3D model
    while(true) {
        if(CheckNewPointCloud()){
            GeneratePointCloud();
            ++mNCurrentFusedFrame;
        }
        else{
//            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if(isStop())
                break;
        }

        if(isStop())
            break;
    }
}
