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
#include <thread>
#include <chrono>
#include <math.h>
#include <pcl/visualization/cloud_viewer.h>
#include "util.h"
#include "Frame.h"
#include "stereo.h"
#include "fusion.h"
#include "pointCloudBuilder.h"

int main(int argc, char **argv) {
    // check for proper command-line usage
    if (argc != 17) {
        fprintf(stderr, "Error: usage %s "
				"<scene> "
				"<intrinsics-file> "
				"<poses-file> "
				"<data-path> "
				"<output-path> "
				"<num-views> "
				"<pre-fusion-thresh> "
				"<post-fusion-thresh> "
				"<support-ratio> "
				"<mapping-file> "
				"<output-pcl-filename> "
				"<alignment-threshold>"
				"<num-disparity>"
				"<window-size>"
				"<num-feature-matches>"
				"<stereo-method>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // read in command-line args
    string dataset_name = argv[1]; // scene
    string intrinsics_file = argv[2]; // intrinsics file
    string pose_file = argv[3]; // intrinsics file
    string data_path = argv[4]; // path to the dataset root directory
    string output_path = argv[5]; // path to all output data
    int num_views = atoi(argv[6]); // number of views used in fusion
    float conf_pre_filt = atof(argv[7]); // pre-filter confidence value
    float conf_post_filt = atof(argv[8]); // post-filter confidence value
    float support_ratio = atof(argv[9]); // support ratio for fusion
    string mapping_file = argv[10]; // path to file mapping timestamp -> image name
    string output_pcl_filename = argv[11]; // output filename for the pointcloud
	float alignment_th = atof(argv[12]); // alignment threshold
	int ndisp = atoi(argv[13]);
	int wsize = atoi(argv[14]);
	int num_matches = atoi(argv[15]);
	string method = argv[16];
	bool post = true;


    scene_name scene;
	if(dataset_name == "STAVRONIKITA") {
    	scene = STAVRONIKITA;
	}
	else if(dataset_name == "PAMIR") {
    	scene = PAMIR;
	}
	else if (dataset_name == "REEF") {
    	scene = REEF;
	}
	else if(dataset_name == "MEXICO") {
        scene = MEXICO;
	}
	else if(dataset_name == "FLORIDA") {
        scene = FLORIDA;
	}
	else {
		cout << "Unsupported scene " << dataset_name << " specified." << endl;
		exit(EXIT_FAILURE);
	}

    // string formatting to add '/' to data_path if it is missing from the input
    size_t str_len = data_path.length();
    if (data_path[str_len-1] != '/') {
        data_path += "/";
    }

    vector<Mat> depth_maps;
    vector<Mat> depth_maps_gt;
    vector<Mat> conf_maps;
    vector<Mat> color_maps;
    vector<Mat> K;
    vector<Mat> D;
    vector<Mat> P;
	Mat R;
	Mat t;
    float baseline;
    cv::Size img_size;

	// define hard-coded variables
	string pose_format = "tum";

	string left_img_dir = data_path + "left/";
	string right_img_dir = data_path + "right/";

	string left_rect_dir = output_path + "rectified_images/left/";
	string right_rect_dir = output_path + "rectified_images/right/";
	string disparity_dir = output_path + "disp_maps/";
	string depth_dir = output_path + "depth_maps/";
	string conf_dir = output_path + "conf_maps/";

	string fusion_dir = output_path + "fused_maps/";
	string fusion_display_dir = fusion_dir + "display/";
	string points_dir = output_path + "point_clouds/";
	string matching_dir = output_path+"feature_matching/";
	string avg_offset_file = output_path+"log/avg_offsets.csv";
	string std_dev_file = output_path+"log/std_dev.csv";
	string xy_offset_file = output_path+"log/xy_offsets.csv";
	string radial_offset_file = output_path+"log/radial_offsets.csv";


    /***** LOAD CAMERA PARAMS *****/
	load_camera_params(K, D, P, R, t, baseline, img_size, intrinsics_file, pose_file, data_path, pose_format);
	int width = img_size.width;
	int height = img_size.height;
	int pose_num = P.size();
    cout << "Number of poses : " << pose_num << endl;

    /***** COLLECT IMAGE FILENAMES *****/
    vector<string> filenames;
    read_mapping_file(filenames, mapping_file, scene);
    int image_num = filenames.size();
    cout << "Number of images: " << image_num << endl;

	// ensure same number of poses and images
    assert(pose_num == image_num);

    /***** STEREO RECTIFICATION *****/
	cv::Mat map_xl, map_yl, map_xr, map_yr;
	cv::Mat K_rect;
	compute_rectified_maps(map_xl, map_yl, map_xr, map_yr, K, D, R, t, K_rect, img_size, scene);


	/***** ITERATE THROUGH IMAGES *****/
	// parameters
//    PointCloud::Ptr globalMap(new PointCloud);
    deque<Frame> keyframesCache;
    double max_magnitude_local = -1;
    int keyframe_ind = 0;

	// timing parameters
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	float duration;
    int fusion_count = 0;
    int rts_count = 0;
    int pcl_count = 0;
    int warping_count = 0;
    double total_fused_time = 0.0;
    double total_rts_time = 0.0;
    double total_pcl_time = 0.0;
    double total_warping_time = 0.0;
	float total_time = 0.0;
    int keyframe_count = 0;

    // These two following Mat are used for stereo process (stereoModule->stereoProcessBySAD)
    cv::Mat left_keyframe;
    cv::Mat right_keyframe;

    // These two following Mat are used for per-sequence rectification
    bool bRefinedFlag = false;
    cv::Mat H_l, H_r;

	// These following variables are used in a new thread for accelerating point cloud generation



	int n_fused_frame = image_num - num_views + 1;
    PointCloudBuilder* pPointCloudBuilder = new PointCloudBuilder(n_fused_frame);
	std::thread* pPointCloudGeneration = new thread(&PointCloudBuilder::Run, pPointCloudBuilder);

	std::chrono::time_point<std::chrono::high_resolution_clock> global_start = std::chrono::high_resolution_clock::now();
    for(int ind  = 0; ind < image_num; ++ind) {
        cout << "\n--------------------------------------" << "Frame " << ind << "---------------------------------------------" << endl;
		/***** RECTIFY STEREO PAIRS *****/
		string left_filename = filenames[ind];
		string right_filename = filenames[ind];

        //Get original input images that are unrectified (images are extracted from rosbag)
        cv::Mat left_img, right_img;
		load_stereo_pair(left_img, right_img, left_img_dir, right_img_dir, left_filename, right_filename);

        // Warp image for stereo rectification
        cv::Mat left_img_rect;
		cv::Mat right_img_rect;
		cv::remap(left_img, left_img_rect, map_xl, map_yl, cv::INTER_LINEAR);
		cv::remap(right_img, right_img_rect, map_xr, map_yr, cv::INTER_LINEAR);


		/***** Post-Rectification Alignment *****/
		/// timing ///
    	start = std::chrono::high_resolution_clock::now();
		/// timing ///

        if(!bRefinedFlag) {
            cv::Mat matching_img;
            std::vector<Point2f> left_feature_points, right_feature_points;
            int ret_val;
            int max_feats = 1000;
            ret_val = match_features(left_img_rect, right_img_rect, max_feats, num_matches, alignment_th, matching_img, left_feature_points, right_feature_points);
            if(ret_val == -1) {
                keyframesCache.clear();
                continue;
            }
            else {
                // compute matching alignment score
		        size_t num_points = left_feature_points.size();
                vector<vector<cv::Point2f> > inliners(num_points, vector<cv::Point2f>(2));
                for(int i = 0; i < num_points; ++i) {
                    inliners[i][0] = cv::Point2f(left_feature_points[i].x, left_feature_points[i].y);
                    inliners[i][1] = cv::Point2f(right_feature_points[i].x, right_feature_points[i].y);
                }
                computeHomography(inliners, H_l, H_r);
                K_rect = H_l * K_rect;
				cout<<"K_rect: "<<K_rect<<endl;
                bRefinedFlag = true;
            }
        }

        // warp right image to better align with left image
        warpPerspective(left_img_rect, left_img_rect, H_l, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        warpPerspective(right_img_rect, right_img_rect, H_r, img_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0); 

		
		/// timing ///
		end = std::chrono::high_resolution_clock::now();
		duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
		total_warping_time += duration;
		cout << "Duration of Post-Rectification Alignment: " << duration << "ms" << endl;
		++warping_count;
		/// timing ///

		/***** KEYFRAME SELCECTION *****/
        // convert left image to grayscale for keyframe evaluation
        cv::Mat left_img_gray;
        cvtColor(left_img_rect, left_img_gray, COLOR_BGR2GRAY);

        // Gradient Computation
        cout << "Current frame: " << left_filename << endl;
		keyframe_ind = ind;
		left_keyframe = left_img_rect.clone();
		right_keyframe = right_img_rect.clone();

		// Select frame with max gradient maginitude as keyframe
		cout << "Keyframe id: " << keyframe_ind << endl;
		string keyframe_filename = filenames[keyframe_ind];

		// store rectified keyframe images
		string left_rect_file = left_rect_dir + keyframe_filename;
		string right_rect_file = right_rect_dir + keyframe_filename;
		cv::imwrite(left_rect_file, left_keyframe);
		cv::imwrite(right_rect_file, right_keyframe);

		// convert to grayscale
		cv::Mat left_keyframe_gray, right_keyframe_gray;
		cvtColor(left_keyframe, left_keyframe_gray, COLOR_BGR2GRAY);
		cvtColor(right_keyframe, right_keyframe_gray, COLOR_BGR2GRAY);

		/***** Real Time Stereo *****/
		/** timing **/
    	start = std::chrono::high_resolution_clock::now();
		/** timing **/

		cv::Mat disp;
		cv::Mat conf;
		cv::Mat cost1;
		cv::Mat cost2;
		cv::Mat cost3;

		runStereo(ndisp, wsize, post, method, keyframe_filename, left_keyframe_gray, right_keyframe_gray, disp, conf, cost1, cost2, cost3);

		/** timing **/
		end = std::chrono::high_resolution_clock::now();
		duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
		total_rts_time += duration;
		cout << "Duration of Real-Time-Stereo: " << duration << "ms" << endl;
		++rts_count;
		/** timing **/

		// display maps
		string disparity_file = disparity_dir + keyframe_filename;
		display_depth(disp.clone(), disparity_file, 100);

		string conf_file = conf_dir + keyframe_filename;
		display_depth(conf.clone(), conf_file, 1);

		// Create a keyframe
		Frame curr_keyframe = Frame(K_rect, baseline, keyframe_ind, img_size);
		curr_keyframe.setPose(P[keyframe_ind]);
		curr_keyframe.setColorMap(left_keyframe);
		curr_keyframe.mConfMap = conf.clone();

		// Create a depth map with subpixel for current keyframe
		curr_keyframe.parabola_fitting(cost1, cost2, cost3, disp);

		// display depth maps
		string depth_file = depth_dir + keyframe_filename;
		save_depth_map(depth_dir, keyframe_filename, curr_keyframe.mDepthMap.clone());
		// display_depth(curr_keyframe.mDepthMap.clone(), depth_file, 60*baseline);

		// release cost mats
		free(cost1.data);
		free(cost2.data);
		free(cost3.data);

		// Add new keyframe
		keyframesCache.push_back(curr_keyframe);

		// Rest parameter
		max_magnitude_local = -1;
		left_keyframe.release();
		right_keyframe.release();

        /***** DEPTH MAP FUSION *****/
		int cache_size = keyframesCache.size();
        if (cache_size == num_views) {
            ++keyframe_count;

            cv::Mat fused_map = Mat::zeros(img_size, CV_32F);
            cv::Mat fused_conf = Mat::zeros(img_size, CV_32F);

            int localRefIndex = (int) num_views / 2;

			// display which frames we are fusing
			cout << "Fusing frames [";
			for (int n=0; n < num_views-1; ++n) {
				cout << keyframesCache.at(n).mIndex << ", ";
			}
			cout << keyframesCache.at((num_views-1)).mIndex << "]..." << endl;

            Frame ReferenceFrame = keyframesCache.at(localRefIndex);

			/** timing **/
            start = std::chrono::high_resolution_clock::now();
			/** timing **/

			//fused_map = ReferenceFrame.mDepthMap;
			//fused_conf = ReferenceFrame.mConfMap;
            confidence_fusion(
                    img_size,
                    fused_map,
                    fused_conf,
                    K_rect,
                    P,
                    keyframesCache,
                    ReferenceFrame.mIndex, //current frame's frame id
                    conf_pre_filt,
                    conf_post_filt,
                    support_ratio);

			/** timing **/
            end = std::chrono::high_resolution_clock::now();
			duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            total_fused_time += duration;
            cout << "Duration of fusion: " << duration << "ms" << endl;
            ++fusion_count;
			/** timing **/

			// display output
			string ref_filename = filenames[ReferenceFrame.mIndex];
            save_depth_map(fusion_dir, ref_filename, fused_map);
			display_depth(fused_map.clone(), fusion_display_dir + ref_filename, 60*baseline);

            string pcl_path = points_dir + ref_filename;
            pPointCloudBuilder->InsertPointCloud(fused_map, ReferenceFrame, pcl_path);
            ++pcl_count;
            // (*globalMap) += *cloud; // Add current 3D model into whole model

            // Pop out earliest Frame.
            keyframesCache.pop_front();
        }

		// release mats
		free(disp.data);
		free(conf.data);
    }
	// pc_thread.join();
	std::chrono::time_point<std::chrono::high_resolution_clock> global_end = std::chrono::high_resolution_clock::now();
	total_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start).count())/1000;
    cout << "=======================================================================================================" << endl;
	cout << "Total time of pipeline: " << total_time/1000 << "s" << endl;
	cout << "Real time performance of pipeline: " << 1000*image_num/total_time<<" frames/s" << endl;
    cout << "Average time of Post-Rectification Alignment: " << total_warping_time/warping_count << "ms" << endl;
    cout << "Average time of RTS: " << total_rts_time/rts_count << "ms" << endl;
    cout << "Average time of fusion: " << total_fused_time/fusion_count << "ms" << endl;
    cout << "Average time for point cloud generation: " << pPointCloudBuilder->mTotal_pcl_time/pcl_count << "ms" << endl;
    cout << "Number of keyframes: " << keyframe_count << endl;
	
    pcl::io::savePLYFileASCII(output_pcl_filename, *(pPointCloudBuilder->globalMap));
    cout << "Final point cloud saved to: " << output_pcl_filename << endl;

    return EXIT_SUCCESS;
}
