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
#include "fusion.h"
#include "pointCloudBuilder.h"
#include "planesweep.h"

int main(int argc, char **argv) {
    // check for proper command-line usage
    if (argc != 11) {
        fprintf(stderr, "Error: usage %s "
                        "<scene> "
                        "<intrinsics-file> "
                        "<poses-file> "
                        "<data-path> "
                        "<output-path> "
                        "<num-views> "
                        "<pre-fusion-thresh> "
                        "<post-fusion-thresh> "
                        "<support-ratio>"
                        "<mapfile>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // read in command-line args
    string dataset_name = argv[1]; // scene
    string intrinsics_file = argv[2]; // intrinsics file
    string pose_file = argv[3]; // pose file
    string data_path = argv[4]; // path to the dataset root directory
    string output_path = argv[5]; // path to all output data
    int num_views = atoi(argv[6]); // number of views used in fusion
    float conf_pre_filt = atof(argv[7]); // pre-filter confidence value
    float conf_post_filt = atof(argv[8]); // post-filter confidence value
    float support_ratio = atof(argv[9]); // support ratio for fusion
    string mapping_file = argv[10]; // path to file mapping timestamp -> image name
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

    string plane_sweep_dir = output_path + "plane_sweep/";

    /***** LOAD CAMERA PARAMS *****/
    load_camera_params(K, D, P, R, t, baseline, img_size, intrinsics_file, pose_file, data_path, pose_format);
    cv::Mat K_left = K[0];


    cv::Mat K_left_inv = K_left.inv();

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


    // plane sweep
    float min_depth=0.1;
    float max_depth=35*baseline;
    int num_planes=100;
    std::vector<double> plane_normal{0,0,1};
    PlaneSweep *ps = new PlaneSweep(width, height);
    ps->imgs.resize(image_num);
    ps->configPlanes( min_depth,max_depth, num_planes);

    /***** ITERATE THROUGH IMAGES *****/
    deque<Frame> keyframesCache;
    // timing parameters
    float total_time = 0.0;
    int keyframe_count = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    float duration;

    // image_num = 10;
    std::chrono::time_point<std::chrono::high_resolution_clock> global_start = std::chrono::high_resolution_clock::now();
    for(int ind  = 0; ind < image_num; ++ind) {
        cout << "\n--------------------------------------" << "Frame " << ind << "---------------------------------------------" << endl;
        string left_filename = filenames[ind];
        string left_img_file = left_img_dir+left_filename;

        cv::Mat left_img = cv::imread(left_img_file, cv::IMREAD_GRAYSCALE);
        if(left_img.empty()) {
            cerr << endl << "Failed to load image at: "
                 << left_img_file << endl;
            exit(EXIT_FAILURE);
        }


        // loadImagesToMemory
        ps->imgs[ind] = left_img;

        // Create a keyframe
        Frame curr_keyframe = Frame(K_left, baseline, ind, img_size);
        curr_keyframe.setPose(P[ind]);
        // Add new keyframe
        keyframesCache.push_back(curr_keyframe);

        /***** DEPTH MAP FUSION *****/
        int cache_size = keyframesCache.size();
        if (cache_size == num_views) {
            ++keyframe_count;

            // apply plane sweep
            start = std::chrono::high_resolution_clock::now();
            ps->sweep(keyframesCache, plane_normal);
            end = std::chrono::high_resolution_clock::now();
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            cout << "Duration of plane sweep: " << duration/1000 << " s" << endl;

            int localRefIndex = (int) num_views / 2;
            Frame ReferenceFrame = keyframesCache.at(localRefIndex);

            //save detph map
            string ref_filename = filenames[ReferenceFrame.mIndex];
            save_depth_map(plane_sweep_dir, ref_filename, ps->depth_map);
            // Pop out earliest Frame.
            keyframesCache.pop_front();
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> global_end = std::chrono::high_resolution_clock::now();
    total_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start).count())/1000;
    cout << "=======================================================================================================" << endl;
    cout << "Total time of pipeline: " << total_time/1000 << "s" << endl;
    cout << "Number of keyframes: " << keyframe_count << endl;


    return EXIT_SUCCESS;
}