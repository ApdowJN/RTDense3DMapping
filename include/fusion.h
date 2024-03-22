

void confidence_fusion(const cv::Size size, cv::Mat &fused_map, cv::Mat &fused_conf, const cv::Mat &K, const vector<Mat> &P, const deque<Frame> &views, const long unsigned int &sequenceId, const float &conf_pre_filt, const float &conf_post_filt, const float &support_ratio);
