#ifndef _UTIL_H_
#define _UTIL_H_

#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "Frame.h"
using namespace std;
using namespace cv;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;


enum scene_name{
    STAVRONIKITA=0,
	PAMIR=1,
    REEF=2,
    MEXICO=3,
    FLORIDA=4,
};

// In-Lines
inline bool str_comp(char *a, char *b) {
    int res = strcmp(a,b);
    bool ret_val;

    if (res < 0) {
        ret_val = true;
    } else {
        ret_val = false;
    }

    return ret_val;
}

inline void pad(std::string &str, const int width, const char c) {
    str.insert(str.begin(), width-str.length(), c);
}


// Utilities
float l2_norm(float x1, float y1, float x2, float y2);
bool isFlyingPoint(const Mat &patch, int filter_width, int num_inliers);
bool matIsEqual(const cv::Mat &mat1, const cv::Mat &mat2);


// Visualization
void display_depth(const Mat &map, string filename, const float &max);


// IO
void load_camera_params(vector<Mat> &K, vector<Mat> &D, vector<Mat> &P, Mat &R, Mat &t, float &baseline, Size &size, const string &intrinic_path, const string &pose_path, const string &data_path, const string &pose_format);
Mat load_pfm(const string filePath);
void load_stereo_pair(cv::Mat &left_img, cv::Mat &right_img, const string &left_img_dir, const string &right_img_dir, const string &left_filename, const string &right_filename);
void read_mapping_file(vector<string> &filenames, const string& file_path, const scene_name& scene);
void write_ply(const Mat &depth_map, const Mat &K, const Mat &P, const string filename, const Mat &rgb_img);
void write_pfm(const cv::Mat image,  const char* filename);
void save_depth_map(const string &saveFolder, const string &frameName, const cv::Mat &depth_map);
void write_txt(const string &filePath);


// Conversion
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );
Mat convert_to_CvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
cv::Mat toCvSE3(const Eigen::Isometry3d &T);


// Image Manipulation
void downsample(cv::Mat &left_img, cv::Mat &right_img, cv::Mat & K, const cv::Size new_size);
void warpImages(const vector<vector<cv::Point2f>> &inliers, cv::Mat &inLeftImg, cv::Mat &inRightImg, cv::Mat &outLeftImg, cv::Mat &outRightImg);
void computeHomography(const vector<vector<cv::Point2f>> &inliers, cv::Mat &H_l, cv::Mat &H_r);
void compute_rectified_maps(cv::Mat &map_xl, cv::Mat &map_yl, cv::Mat &map_xr, cv::Mat &map_yr, const vector<cv::Mat> &K, const vector<cv::Mat> &D, const cv::Mat &R, const cv::Mat &t, cv::Mat &K_rect, const cv::Size &size, const scene_name &scene);
int ExtractLineSegment(const Mat &img, const Mat &image2, vector<cv::line_descriptor::KeyLine> &keylines,vector<cv::line_descriptor::KeyLine> &keylines2);
float med_filt(const Mat &patch, int filter_width, int num_inliers);
float mean_filt(const Mat &patch, int filter_width, int num_inliers);
double sobel_filter(cv::Mat& img_gray);
int match_features(const cv::Mat &img_1, const cv::Mat &img_2, const int &max_features, const int &num_matches, const int &max_y_offset, cv::Mat &matching_img, std::vector<Point2f> &feature_points_1, std::vector<Point2f> &feature_points_2);


// Point Cloud Manipulation
vector<int> searchNearestPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target, const int binSize, int flag);
unordered_set<int> groupPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target);
pcl::PointCloud<pcl::PointXYZ>::Ptr read_txt(const string &filePath);
PointCloud::Ptr generated_point_clouds(const cv::Mat &imLeft, const cv::Mat &imDepth, const Mat &Tcw,  const Mat &K);


#endif
