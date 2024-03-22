#include <stdio.h>
#include <vector>
#include <mutex>
#include <dirent.h>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstring>

#include "util.h"


/********** UTILITIES **********/
float l2_norm(float x1, float y1, float x2, float y2) {
	double x = x1 - x2;
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);
	dist = sqrt(dist);

	return dist;
}

bool isFlyingPoint(const Mat &patch, int filter_width, int num_inliers) {
    int inliers = 0;
    for (int r=0; r<filter_width; ++r) {
        for (int c=0; c<filter_width; ++c) {
            if (patch.at<float>(r,c) > 0) {
                ++inliers;
            }
        }
    }
    return (inliers <= num_inliers) ? true: false;

}

bool matIsEqual(const cv::Mat &mat1, const cv::Mat &mat2) {
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims||
        mat1.channels()!=mat2.channels()) {
        return false;
    }
    if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
        return false;
    }
    size_t nrOfElements1 = mat1.total()*mat1.elemSize();
    if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;
    bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
    return lvRet;
}

/********** VISUALIZATION **********/
void display_depth(const Mat &map, string filename, const float &max) {
    Size size = map.size();
	int crop_offset = 0;
    Mat cropped = map(Rect(crop_offset,crop_offset,size.width-((crop_offset+1)*2),size.height-((crop_offset+1)*2)));
    cropped = (cropped * 255) / max;
    Mat output;
    threshold(cropped, output, 0, 255, THRESH_TOZERO);
    imwrite(filename, output);
}

/********** IO **********/
void load_camera_params(vector<Mat> &K, vector<Mat> &D, vector<Mat> &P, Mat &R, Mat &t, float &baseline, Size &size, const string &intrinsic_path, const string &pose_path, const string &data_path, const string &pose_format) {

	// Load camera intrinsics
	cout << intrinsic_path << endl;
    cv::FileStorage intrins_file(intrinsic_path, cv::FileStorage::READ);
    if(!intrins_file.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return;
    }

	// extract info for left camera
	FileNode left_cam = intrins_file["left"];
	cv::Mat left_K;
	left_cam["K"] >> left_K;
	left_K.convertTo(left_K, CV_32F);
	K.push_back(left_K);

	cv::Mat left_D;
	left_cam["D"] >> left_D;
	left_D.convertTo(left_D, CV_32F);
	D.push_back(left_D.t());

	// extract info for right camera
	FileNode right_cam = intrins_file["right"];
	cv::Mat right_K;
	right_cam["K"] >> right_K;
	right_K.convertTo(right_K, CV_32F);
	K.push_back(right_K);

	cv::Mat right_D;
	right_cam["D"] >> right_D;
	right_D.convertTo(right_D, CV_32F);
	D.push_back(right_D.t());

	// extract baseline, size, rotation, and translation info
	// intrins_file["baseline"] >> baseline;
   
	// convert baseline to meters (assume it is in mm if > 1)
	// if(baseline > 1) {
	// 	baseline /= 1000;
	// }

	intrins_file["size"] >> size;
	intrins_file["R"] >> R;
	intrins_file["T"] >> t;
    baseline = cv::norm(t, cv::NORM_L2);
	t = t.t();

	intrins_file.release();

	// Load camera pose info
    cout << "Loading Camera Poses from Underwater Dataset ..." << endl;
    int count = 1;

    //read pose from underwater
    ifstream fp;
    fp.open(pose_path);
    cout<<"Load poses from "<<pose_path<<endl;

    while(!fp.eof()) {
        string s;
        getline(fp,s);
        if(!s.empty()) {
            int found1 = s.find(" ");
            string head = s.substr(0, found1);
            string after_head = s.substr(found1+1);
            if(head == "frame" || head == "stamp")
                continue;
            string token;
            string delimiter = " ";
            Eigen::Quaterniond q(0,0,0,0);
            Eigen::Vector3d t;
            stringstream ss(after_head);
            string pi;
            int index = 0;

			if (pose_format == "tum") {
				while(getline(ss, pi, ' ')) {
					if(index == 0)
						t(0) = stod(pi);
					else if(index == 1)
						t(1) = stod(pi);
					else if(index == 2)
						t(2) = stod(pi);
					else if(index == 3)
						q.x() = stod(pi);
					else if(index == 4)
						q.y() = stod(pi);
					else if(index == 5)
						q.z() = stod(pi);
					else if(index == 6)
						q.w() = stod(pi);
					++index;
				}
			}
			else {
				while(getline(ss, pi, ' ')) {
				    if(index == 0)
				        q.x() = stod(pi);
				    else if(index == 1)
				        q.y() = stod(pi);
				    else if(index == 2)
				        q.z() = stod(pi);
				    else if(index == 3)
				        q.w() = stod(pi);
				    else if (index == 4)
				        t(0) = stod(pi);
				    else if (index == 5)
				        t(1) = stod(pi);
				    else if (index == 6)
				        t(2) = stod(pi);
				    ++index;
				}
			}

            Eigen::Matrix3d Rwc = q.toRotationMatrix();
            Eigen::Matrix3d Rcw = Rwc.transpose();
            Eigen::Vector3d tcw = -Rcw*t;
            cv::Mat P_i = toCvSE3(Rcw, tcw);
            P.push_back(P_i);
            ++count;
        }
    }
    fp.close();
}

void load_stereo_pair(cv::Mat &left_img, cv::Mat &right_img, const string &left_img_dir, const string &right_img_dir, const string &left_filename, const string &right_filename) {
	string left_img_file = left_img_dir+left_filename;
	string right_img_file = right_img_dir+right_filename;
	left_img = cv::imread(left_img_file, cv::IMREAD_UNCHANGED);
	right_img = cv::imread(right_img_file, cv::IMREAD_UNCHANGED);

	if(left_img.empty()) {
		cerr << endl << "Failed to load image at: "
			 << left_img_file << endl;
		exit(EXIT_FAILURE);
	}

	if(right_img.empty()) {
		cerr << endl << "Failed to load image at: "
			 << right_img_file << endl;
		exit(EXIT_FAILURE);
	}
}

Mat load_pfm(const string filePath)
{

    //Open binary file
    ifstream file(filePath.c_str(),  ios::in | ios::binary);

    Mat imagePFM;

    //If file correctly openened
    if(file)
    {
        //Read the type of file plus the 0x0a UNIX return character at the end
        char type[3];
        file.read(type, 3*sizeof(char));

        //Read the width and height
        unsigned int width(0), height(0);
        file >> width >> height;

        //Read the 0x0a UNIX return character at the end
        char endOfLine;
        file.read(&endOfLine, sizeof(char));

        int numberOfComponents(0);
        //The type gets the number of color channels
        if(type[1] == 'F')
        {
            imagePFM = Mat(height, width, CV_32FC3);
            numberOfComponents = 3;
        }
        else if(type[1] == 'f')
        {
            imagePFM = Mat(height, width, CV_32FC1);
            numberOfComponents = 1;
        }

        //Read the endianness plus the 0x0a UNIX return character at the end
        //Byte Order contains -1.0 or 1.0
        char byteOrder[4];
        file.read(byteOrder, 4*sizeof(char));

        //Find the last line return 0x0a before the pixels of the image
        char findReturn = ' ';
        while(findReturn != 0x0a)
        {
          file.read(&findReturn, sizeof(char));
        }

        //Read each RGB colors as 3 floats and store it in the image.
        float *color = new float[numberOfComponents];
        for(unsigned int i = 0 ; i<height ; ++i)
        {
            for(unsigned int j = 0 ; j<width ; ++j)
            {
                file.read((char*) color, numberOfComponents*sizeof(float));

                //In the PFM format the image is upside down
                if(numberOfComponents == 3)
                {
                    //OpenCV stores the color as BGR
                    imagePFM.at<Vec3f>(height-1-i,j) = Vec3f(color[2], color[1], color[0]);
                }
                else if(numberOfComponents == 1)
                {
                    //OpenCV stores the color as BGR
                    imagePFM.at<float>(height-1-i,j) = color[0];
                }
            }
        }

        delete[] color;

        //Close file
        file.close();
    }
    else
    {
        cerr << "Could not open the file : " << filePath << endl;
    }

    return imagePFM;
}

void read_mapping_file(vector<string> &filenames, const string& file_path, const scene_name& scene)
{
    char separator = ',';
    string row, item;

    if(scene == STAVRONIKITA || scene == PAMIR || scene == REEF || scene == MEXICO || scene == FLORIDA) {
        ifstream in(file_path.c_str());
        while(getline(in, row)) {
            stringstream ss(row);
            int i = 0;
            while (getline(ss, item, separator))
                i++;
            if (i % 2 == 0) {
                filenames.push_back(item);
            }
        }
        in.close();
    }
	else {
		cout << "Reading mapping file for this scene is not yet supported" << endl;
		exit(EXIT_FAILURE);
	}
}

void write_ply(const Mat &depth_map, const Mat &K, const Mat &P, const string filename, const Mat &rgb_img) {
    Size size = depth_map.size();

    int rows = size.height;
    int cols = size.width;

    vector<Mat> ply_points;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<cols; ++c) {
            float depth = depth_map.at<float>(r,c);
//            size_t depth_i = depth_map.at<ushort>(r,c);
//            float depth = depth_i*1.0;
            if (depth <= 0) {
                continue;
            }

            //depth = depth/100;
            if(depth >=400 || depth <=4)
                continue;
            // compute corresponding (x,y) locations
            Mat x_1(4,1,CV_32F);
            x_1.at<float>(0,0) = depth * c;
            x_1.at<float>(1,0) = depth * r;
            x_1.at<float>(2,0) = depth;
            x_1.at<float>(3,0) = 1;

            // find 3D world coord of back projection
            Mat cam_coords = K.inv() * x_1;
            Mat X_world = P.inv() * cam_coords;
            X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
            X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
            X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
            X_world.at<float>(0,3) = 1.0;

            Mat ply_point = Mat::zeros(1,6,CV_32F);

            ply_point.at<float>(0,0) = X_world.at<float>(0,0);
            ply_point.at<float>(0,1) = X_world.at<float>(0,1);
            ply_point.at<float>(0,2) = X_world.at<float>(0,2);

            Vec3b intensity = rgb_img.at<Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            ply_point.at<float>(0,3) = red;
            ply_point.at<float>(0,4) = green;
            ply_point.at<float>(0,5) = blue;

            ply_points.push_back(ply_point);
        }
    }

    ofstream ply_file;
    ply_file.open(filename);
    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << ply_points.size() << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "element face 0\n";
    ply_file << "end_header\n";

    vector<Mat>::iterator pt(ply_points.begin());
    for (; pt != ply_points.end(); ++pt) {
        //cout<< pt->at<float>(0,0) << " " << pt->at<float>(0,1) << " " << pt->at<float>(0,2) << " " << pt->at<float>(0,3) << " " << pt->at<float>(0,4) << " " << pt->at<float>(0,5)<<endl;
        ply_file << pt->at<float>(0,0) << " " << pt->at<float>(0,1) << " " << pt->at<float>(0,2) << " " << pt->at<float>(0,3) << " " << pt->at<float>(0,4) << " " << pt->at<float>(0,5) << "\n";
    }

    ply_file.close();

}

void write_pfm(const cv::Mat image, const char* filename) {
    int width = image.cols;
    int height = image.rows;

    float* depths=(float*)malloc(width*height*sizeof(float));
    std::fill_n(depths,width*height, 0.0);
    for (int vL = 0; vL < height; ++vL) {
        for (int uL = 0; uL < width; ++uL) {

            float depth = image.at<float>(vL, uL);

//            if (depth <= 0 || depth >= 5.0)
//                continue;

            depths[vL*width+uL] = depth;
        }
    }

    std::ofstream ofs(filename, std::ifstream::binary);
    ofs << "Pf\n" << width << " " << height << "\n"<<-1.0<< "\n";
    float* tbimg = (float *)malloc(width*height*sizeof(float));
    //PFM SPEC image stored bottom -> top reversing image
#pragma omp parallel
    {
#pragma omp for
        for(int i =0; i< height; i++){
            memcpy(&tbimg[(height -i-1)*width],&depths[(i*width)],width*sizeof(float));
        }
    }

    ofs.write(( char *)tbimg,width*height*sizeof(float));
    ofs.close();
    free(tbimg);
	free(depths);
}

void save_depth_map(const string &saveFolder, const string &frameName, const cv::Mat &depth_map) {
    string save_fused_depth_path = saveFolder + frameName;
    int len = save_fused_depth_path.size();
    len -= 4; //remove .png
    string token = save_fused_depth_path.substr(0, len);
    save_fused_depth_path = token + ".pfm";
    write_pfm(depth_map, save_fused_depth_path.c_str());
}

void write_text(const string &filePath, const vector<float> &vPointKNNDistance ) {
    std::ofstream ofile;
    ofile.open(filePath, std::ios::app);

    for (size_t i = 0; i < vPointKNNDistance.size(); ++i) {
        //cout<< pt->at<float>(0,0) << " " << pt->at<float>(0,1) << " " << pt->at<float>(0,2) << " " << pt->at<float>(0,3) << " " << pt->at<float>(0,4) << " " << pt->at<float>(0,5)<<endl;
        ofile << vPointKNNDistance[i]<<", ";
    }
    ofile<<"\n";
    ofile.close();
}

/********** CONVERSION **********/
Mat convert_to_CvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Isometry3d cvMat2Eigen( cv::Mat& R, cv::Mat& tvec )
{
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ )
            r(i,j) = R.at<double>(i,j);

    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;

    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}


/********** IMAGE MANIPULATION **********/
void downsample(cv::Mat &left_img, cv::Mat &right_img, cv::Mat & K, const cv::Size new_size) {
	cout << "Downsampling not yet supported..." << endl;
	// Downsample image if their resolution are 1600*1200
	//if (left_img_rect.rows == 1200 && left_img_rect.cols == 1600) {
	//	cv::Mat left_down, right_down;
	//	cv::Size sz(800,600);
	//	cv::resize(left_img_rect, left_down, sz, 0, 0, cv::INTER_LINEAR);
	//	cv::resize(right_img_rect, right_down, sz, 0, 0, cv::INTER_LINEAR);

	//	left_img_rect = left_down;
	//	right_img_rect = right_down;
	//}
}

void warpImages(const vector<vector<cv::Point2f>> &inliers, cv::Mat &inLeftImg, cv::Mat &inRightImg, cv::Mat &outLeftImg, cv::Mat &outRightImg)
{

    int num_points = inliers.size();
    Mat A_l = cv::Mat::ones(num_points, 2, CV_32F);
    Mat A_r = cv::Mat::ones(num_points, 2, CV_32F);
    Mat b_l = cv::Mat::ones(num_points, 1, CV_32F);
    Mat b_r = cv::Mat::ones(num_points, 1, CV_32F);
    Mat x_l, x_r;

    float y_avg, y_left, y_right;
    for (int i = 0; i < num_points; i++){
        y_left = inliers[i][0].y;
        y_right = inliers[i][1].y;
        y_avg = (y_left + y_right)/2.0;

        A_l.at<float>(i,0) = inliers[i][0].y;
        b_l.at<float>(i,0) = y_avg;

        A_r.at<float>(i,0) = inliers[i][1].y;
        b_r.at<float>(i,0) = y_avg;
    }
    solve(A_l, b_l, x_l, DECOMP_SVD);
    solve(A_r, b_r, x_r, DECOMP_SVD);

    //build homography matrices
    Mat H_l = cv::Mat::zeros(3,3, CV_32F);
    Mat H_r = cv::Mat::zeros(3,3, CV_32F);
    H_l.at<float>(0,0) = 1;
    H_r.at<float>(0,0) = 1;
    H_l.at<float>(2,2) = 1;
    H_r.at<float>(2,2) = 1;

    H_l.at<float>(1,1) = x_l.at<float>(0,0);
    H_l.at<float>(1,2) = x_l.at<float>(1,0);

    H_r.at<float>(1,1) = x_r.at<float>(0,0);
    H_r.at<float>(1,2) = x_r.at<float>(1,0);

    cout<<"homography matrix for left image: "<<H_l<<endl;
    cout<<"homography matrix for right image: "<<H_r<<endl;
    Size dsize = inLeftImg.size();

    warpPerspective(inLeftImg, outLeftImg, H_l, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP
    warpPerspective(inRightImg, outRightImg, H_r, dsize, INTER_LINEAR , BORDER_CONSTANT, 0); //+ WARP_INVERSE_MAP

}

void computeHomography(const vector<vector<cv::Point2f>> &inliers, cv::Mat &H_l, cv::Mat &H_r)
{
    int num_points = inliers.size();
    Mat A_l = cv::Mat::ones(num_points, 2, CV_32F);
    Mat A_r = cv::Mat::ones(num_points, 2, CV_32F);
    Mat b_l = cv::Mat::ones(num_points, 1, CV_32F);
    Mat b_r = cv::Mat::ones(num_points, 1, CV_32F);
    Mat x_l, x_r;

    float y_avg, y_left, y_right;
    for (int i = 0; i < num_points; i++){
        y_left = inliers[i][0].y;
        y_right = inliers[i][1].y;
        y_avg = (y_left + y_right)/2.0;

        A_l.at<float>(i,0) = inliers[i][0].y;
        b_l.at<float>(i,0) = y_avg;

        A_r.at<float>(i,0) = inliers[i][1].y;
        b_r.at<float>(i,0) = y_avg;
    }
    solve(A_l, b_l, x_l, DECOMP_SVD);
    solve(A_r, b_r, x_r, DECOMP_SVD);

    //build homography matrices
    H_l = cv::Mat::eye(3,3, CV_32F);
    H_r = cv::Mat::eye(3,3, CV_32F);
    
    H_l.at<float>(1,1) = x_l.at<float>(0,0);
    H_l.at<float>(1,2) = x_l.at<float>(1,0);

    H_r.at<float>(1,1) = x_r.at<float>(0,0);
    H_r.at<float>(1,2) = x_r.at<float>(1,0);

    cout<<"homography matrix for left image: "<<H_l<<endl;
    cout<<"homography matrix for right image: "<<H_r<<endl;
}

cv::Mat toCvSE3(const Eigen::Isometry3d &T)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            cvMat.at<float>(i,j)=T(i,j);
        }
    }
    return cvMat.clone();

}

void compute_rectified_maps(cv::Mat &map_xl, cv::Mat &map_yl, cv::Mat &map_xr, cv::Mat &map_yr, const vector<cv::Mat> &K, const vector<cv::Mat> &D, const cv::Mat &R, const cv::Mat &t, cv::Mat &K_rect, const cv::Size &size, const scene_name &scene) {
	cout << "Beginning Rectification..." << endl;
    if (scene == STAVRONIKITA || scene == PAMIR || scene == REEF || scene == MEXICO || scene == FLORIDA) {
        cv::Mat left_R, right_R, left_KP, right_KP, Q;
        cv::Rect validRoi[2];

        cv::stereoRectify(
				K[0],
				D[0],
				K[1],
				D[1],
				size,
				R,
				t,
				left_R,
				right_R,
				left_KP,
				right_KP,
				Q,
				cv::CALIB_ZERO_DISPARITY,
				0,
				size,
				&validRoi[0],
				&validRoi[1]);

		// set rectified intrinsic matrix
		K_rect = left_KP.rowRange(0,3).colRange(0,3);
		K_rect.convertTo(K_rect, CV_32F);

		// compute pixel mappings for rectification
        cv::initUndistortRectifyMap(K[0], D[0], left_R, left_KP.rowRange(0,3).colRange(0,3), size, CV_32F, map_xl, map_yl);
        cv::initUndistortRectifyMap(K[1], D[1], right_R, right_KP.rowRange(0,3).colRange(0,3), size, CV_32F, map_xr, map_yr);
    }
	//else if (scene == MEXICO) {
    //    cv::Mat left_R, right_R, left_KP, right_KP;
    //    fsSettings["LEFT.R"] >>left_R;
    //    fsSettings["RIGHT.R"] >>right_R;
    //    fsSettings["LEFT.P"] >>left_KP;
    //    fsSettings["RIGHT.P"] >>right_KP;

    //    cout << "Left KP: " << left_KP << "\n" << endl;
    //    cout << "Right KP: " << right_KP << "\n" << endl;
    //    cout << "Left R: " << left_R << "\n" << endl;
    //    cout << "Right R: " << right_R << "\n" << endl;

    //    cv::initUndistortRectifyMap(K[0], D[0], left_R, left_KP.rowRange(0,3).colRange(0,3), size, CV_32F, map_x[0], map_y[0]);
    //    cv::initUndistortRectifyMap(K[1], D[1], right_R, right_KP.rowRange(0,3).colRange(0,3), size, CV_32F, map_x[1], map_y[1]);
    //}
	else {
		cout << "Computing rectification maps for this scene is not yet supported" << endl;
		exit(EXIT_FAILURE);
	}
}

int ExtractLineSegment(const Mat &img, const Mat &image2, vector<cv::line_descriptor::KeyLine> &keylines,vector<cv::line_descriptor::KeyLine> &keylines2) {
    cv::Mat mLdesc,mLdesc2;
    vector<vector<cv::DMatch>> lmatches;
    Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();

    lsd->detect(img, keylines, 1.2,1);
    lsd->detect(image2,keylines2,1.2,1);
    int lsdNFeatures = 50;
	int keylines_size = keylines.size();
	int keylines2_size = keylines2.size();
    if(keylines_size>lsdNFeatures) {
        sort(keylines.begin(), keylines.end(),[](const cv::line_descriptor::KeyLine &a,const cv::line_descriptor::KeyLine &b){return a.response > b.response;});
        keylines.resize(lsdNFeatures);
        for( int i=0; i<lsdNFeatures; i++)
            keylines[i].class_id = i;
    }
    if(keylines2_size>lsdNFeatures) {
        sort(keylines2.begin(), keylines2.end(), [](const cv::line_descriptor::KeyLine &a,const cv::line_descriptor::KeyLine &b){return a.response > b.response;});
        keylines2.resize(lsdNFeatures);
        for(int i=0; i<lsdNFeatures; i++)
            keylines2[i].class_id = i;
    }

    lbd->compute(img, keylines, mLdesc);
    lbd->compute(image2,keylines2,mLdesc2);
    cv::BFMatcher* bfm = new cv::BFMatcher(NORM_HAMMING, false);
    bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    vector<DMatch> matches;
    for(size_t i=0;i<lmatches.size();i++)
    {
        const DMatch& bestMatch = lmatches[i][0];
        const DMatch& betterMatch = lmatches[i][1];
        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < 0.7)
            matches.push_back(bestMatch);
    }

    // Draw matching for visulization
//    cv::Mat outImg;
//    std::vector<char> mask( lmatches.size(), 1 );
//    drawLineMatches( img, keylines, image2, keylines2, matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask, cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
//    imshow( "Matches", outImg );
//    waitKey(0);
//    imwrite("Line_Matcher.png",outImg);
    return matches.size();
}

float med_filt(const Mat &patch, int filter_width, int num_inliers) {
    vector<float> vals;
    int inliers = 0;
    float initial_val = patch.at<float>((filter_width-1)/2,(filter_width-1)/2);

    for (int r=0; r<filter_width; ++r) {
        for (int c=0; c<filter_width; ++c) {
            if (patch.at<float>(r,c) > 0) {
                vals.push_back(patch.at<float>(r,c));
                ++inliers;
            }
        }
    }

    sort(vals.begin(), vals.end());
    
    float med;

    if (inliers < num_inliers) {
        med = initial_val;
    } else {
        med = vals[static_cast<int>(inliers/2)];
    }

    return med;
}

float mean_filt(const Mat &patch, int filter_width, int num_inliers) {
    float sum = 0.0;
    int inliers = 0;
    float initial_val = patch.at<float>((filter_width-1)/2,(filter_width-1)/2);

    for (int r=0; r<filter_width; ++r) {
        for (int c=0; c<filter_width; ++c) {
            if (patch.at<float>(r,c) >= 0) {
                sum += patch.at<float>(r,c);
                ++inliers;
            }
        }
    }

    if (inliers < num_inliers) {
        sum = initial_val;
    } else {
        sum /= inliers;
    }

    return sum;
}

double sobel_filter(cv::Mat& img_gray) {
    cv::Size size = img_gray.size();
    int rows = size.height;
    int cols = size.width;
    double total_magnitude = 0.0;

    for(int i = 1; i < rows - 1; i++) {
        for(int j = 1; j < cols - 1; j++) {
            double sum1 =  img_gray.at<uchar>(i-1, j+1) -  img_gray.at<uchar>(i-1, j-1)
                    + 2 * img_gray.at<uchar>(i, j+1) - 2 * img_gray.at<uchar>(i, j-1)
                    + img_gray.at<uchar>(i+1, j+1) - img_gray.at<uchar>(i+1, j-1);


            double sum2 = img_gray.at<uchar>(i+1, j-1) - img_gray.at<uchar>(i-1, j-1)
                    + 2 * img_gray.at<uchar>(i+1, j) - 2 * img_gray.at<uchar>(i-1, j)
                    + img_gray.at<uchar>(i+1, j+1) - img_gray.at<uchar>(i-1, j+1);

            double magnitude =  abs(sum1) + abs(sum2); //approximate magnitude which is mucch faster to compute. G = |Gx| + |Gy|
            total_magnitude += magnitude;
        }
    }

    return total_magnitude;
}

int match_features(const cv::Mat &img_1, const cv::Mat &img_2, const int &max_features, const int &num_matches, const int &max_y_offset, cv::Mat &matching_img, std::vector<Point2f> &feature_points_1, std::vector<Point2f> &feature_points_2) {
	// convert images to grayscale
	cv::Mat img_1_gray, img_2_gray;
	cv::cvtColor(img_1, img_1_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img_2, img_2_gray, cv::COLOR_BGR2GRAY);

	// detect features and compute descriptors
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat left_descriptors, right_descriptors;

	Ptr<Feature2D> orb = cv::ORB::create(max_features);
	orb->detectAndCompute(img_1_gray, cv::Mat(), keypoints_1, left_descriptors);
	orb->detectAndCompute(img_2_gray, cv::Mat(), keypoints_2, right_descriptors);
	
	int k1_size = keypoints_1.size();
	int k2_size = keypoints_2.size();

	if (k1_size < num_matches || k2_size < num_matches) {
		return -1;
	}

	// match features
	std::vector<cv::DMatch> initial_matches;
	std::vector<cv::DMatch> matches;
	Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(left_descriptors, right_descriptors, initial_matches, cv::Mat());

	int matches_size = initial_matches.size();

	if (matches_size < num_matches) {
		return -1;
	}

	// sort matches by matching score
	std::sort(initial_matches.begin(), initial_matches.end());

	// erase poor matches
	initial_matches.erase(initial_matches.begin()+num_matches, initial_matches.end());

	// grab keypoint indices for matches
	//matches_size = matches.size();
	//for (int i=0; i<matches_size; ++i) {
	//	feature_points_1.push_back( keypoints_1[matches[i].queryIdx].pt );
	//	feature_points_2.push_back( keypoints_2[matches[i].trainIdx].pt );
	//}

	// compute matching alignment score
	float xdiff = 0.0;
	float ydiff = 0.0;
	
	for (auto match : initial_matches) {
		Point2f p1 = keypoints_1[match.queryIdx].pt;
		Point2f p2 = keypoints_2[match.trainIdx].pt;
		
		xdiff = (p1.x - p2.x);
		ydiff = std::abs(p1.y - p2.y);


		if (xdiff >= 0 && ydiff <= max_y_offset) {
			matches.push_back(match);
		}
	}

	// Draw top matches
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, matching_img);

	// grab keypoint indices for matches
	matches_size = matches.size();
	for (int i=0; i<matches_size; ++i) {
		feature_points_1.push_back( keypoints_1[matches[i].queryIdx].pt );
		feature_points_2.push_back( keypoints_2[matches[i].trainIdx].pt );
	}

	return 0;
}


/********** POINT CLOUD MANIPULATION **********/
vector<int> searchNearestPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target, const int binSize, int flag) {
    cout<<"start comparsion"<<endl;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(source);
    int K = 1;
    vector<int> vPointIdxKNNSearch;
    vector<float> vPointKNNDistance;

//    pcl::PointCloud<pcl::PointXYZ>::Ptr inliear(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr outliear(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < (int)target->points.size(); ++i) {
        pcl::PointXYZ searchPoint = target->points[i];
        vector<int> tPointIdxKNNSearch;
        vector<float> tPointKNNSquareDistance;

        if (kdtree.nearestKSearch(searchPoint, K, tPointIdxKNNSearch, tPointKNNSquareDistance) > 0) {

            float dis = sqrt(tPointKNNSquareDistance[0]);
//            if (dis > 1)
//                dis = 1.0;
//            if(dis < 0.1 && flag == 1)
//                inliear->points.push_back(searchPoint);
//            else if(dis >= 0.6 && flag == 1)
//                outliear->points.push_back(searchPoint);

            vPointKNNDistance.push_back(dis);
//            minDis = min(dis, minDis);
//            maxDis = max(dis, maxDis);
//            if(i%20000000 == 0)
//                cout<<i<<endl;
        }

    }

    if (flag == 0) {
        cout<<"there are " <<target->points.size()<<" points in sparse colmap"<<endl;
        cout<<"there are " <<source->points.size()<<" points in pipeline"<<endl;
    }
//    if(flag == 1) {
//        pcl::io::savePLYFileASCII("data/inliear_pipeline.ply", *inliear);
//        pcl::io::savePLYFileASCII("data/outliear_pipeline_shipwreck.ply", *outliear);
//    }

    //create histogram
    int interval = 1;//ceil(maxDis - minDis);
    vector<int> disHist(binSize, 0);
    const float factor = binSize/interval;

    for (size_t i = 0; i < vPointKNNDistance.size(); ++i) {
        float distance = vPointKNNDistance[i];
        if(distance > 1)
            distance = 1.0;
        int bin = distance*factor;
//        int bin = round(distance * factor);
        if (bin == binSize)
            bin = binSize-1;

        ++disHist[bin];
    }
//    cout<<minDis<<endl;
//    cout<<maxDis<<endl;
	int disHist_size = disHist.size();
    for(int i = 0; i < disHist_size; i++) {
        cout<<disHist[i]<< ", ";
    }
    cout<<endl;

    if(flag == 0) {
        int num = disHist[0];
        int n_t = target->points.size();
        float percent = num*1.0/n_t;
        cout<<"There are: "<<percent*100<<"% points in COLMAP have within 0.1 m from pipeline"<<endl;
    }
    else if(flag == 1) {
        int num = disHist[0];
        int n_t = target->points.size();
        float percent = num*1.0/n_t;
        cout<<"There are: "<<percent*100<<"% points in Pipeline have within 0.1 m from COLMAP"<<endl;
    }
    int n = vPointKNNDistance.size();
//    vector<double> vPointsqrtDistance(n, 0.0);
//    for (int i =0; i < n; i++) {
//        vPointsqrtDistance[i] = sqrt(vPointKNNDistance[i]);
//    }
    double total_dis = accumulate(vPointKNNDistance.begin(),vPointKNNDistance.end(),0.0);
    cout<<"Average distance: " << total_dis/n<<endl;
    sort(vPointKNNDistance.begin(), vPointKNNDistance.end());
    int idx = n/2;
    cout<<"Median distance: "<<vPointKNNDistance[idx]<<endl;

//    string outPath = "/home/wangweihan/Documents/my_project/underwater_project/code/ICRA/data/histogram.txt";
//    write_text(outPath, vPointKNNDistance);

    return disHist;
}

unordered_set<int> groupPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(source);
    int K = 1;

    unordered_set<int> inliers;
    unordered_set<int> outliers;
    for (int i = 0; i < (int)target->points.size(); ++i) {
        pcl::PointXYZ searchPoint = target->points[i];

        vector<int> tPointIdxKNNSearch;
        vector<float> tPointKNNSquareDistance;
        PointT p;
        if (kdtree.nearestKSearch(searchPoint, K, tPointIdxKNNSearch, tPointKNNSquareDistance) > 0) {

            float dis = sqrt(tPointKNNSquareDistance[0]);
            if (dis > 1)
                dis = 1.0;

            p.x = searchPoint.x;
            p.y = searchPoint.y;
            p.z = searchPoint.z;

            if (dis <= 0.05) {
                inliers.insert(i);

            }

        }

    }

    return inliers;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr read_txt(const string &filePath) {
    ifstream ply_file;
    ply_file.open(filePath);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cout<<"start reading file"<<endl;
    int num = 0;
    while(!ply_file.eof()) {
        string line;
        getline(ply_file, line);
        stringstream ss(line);
        string s;
        int count = 0;
        pcl::PointXYZ pointT;
        while(getline(ss, s, ' ')) {
            if(count == 0)
                pointT.x = stof(s);
            else if(count == 1)
                pointT.y = stof(s);
            else if(count == 2)
                pointT.z = stof(s);
            else {
                cloud->points.push_back(pointT);
                break;
            }
            ++count;
        }
        ++num;
        if(num%10000000 == 0)
            cout<<line<<endl;
//        if (num >=297166)
//            cout<<line<<endl;
    }
    cout<<num<<endl;
    cout<<"finish reading file"<<endl;
    ply_file.close();
    return cloud;
}

PointCloud::Ptr generated_point_clouds(const cv::Mat &imLeft, const cv::Mat &imDepth, const Mat &Tcw,  const Mat &K) {
    cv::Size size = imLeft.size();

    int rows = size.height;
    int cols = size.width;
    int offset = 0;
    PointCloud::Ptr cloud(new PointCloud);
	std::vector<PointT> points;

    cv::Mat color = imLeft;

	// augment K matrix
	cv::Mat K_i = K.t();
	K_i.push_back(Mat::zeros(1,3,CV_32F));
	K_i = K_i.t();
	Mat temp = Mat::zeros(1,4,CV_32F);
	temp.at<float>(0,3) = 1.0;
	K_i.push_back(temp);

	cv::Mat K_inv = K_i.inv();
	cv::Mat P_inv = Tcw.inv();

	K_inv.convertTo(K_inv, CV_32F);
	P_inv.convertTo(P_inv, CV_32F);

#pragma omp parallel num_threads(12)
{
	#pragma omp for collapse(2)
    for (int vL=offset; vL<rows-offset; ++vL) {
        for (int uL=offset; uL<cols-offset; ++uL) {
            float d = imDepth.at<float>(vL, uL);

            if(d <= 0.0)
                continue;

			cv::Mat p_img(4,1,CV_32F);
			p_img.at<float>(0,0) = d*uL;
			p_img.at<float>(1,0) = d*vL;
			p_img.at<float>(2,0) = d;
			p_img.at<float>(3,0) = 1;

			cv::Mat p_2d = K_inv * p_img;

			cv::Mat p_3d = P_inv * p_2d;
			p_3d.at<float>(0,0) = p_3d.at<float>(0,0) / p_3d.at<float>(0,3);
			p_3d.at<float>(0,1) = p_3d.at<float>(0,1) / p_3d.at<float>(0,3);
			p_3d.at<float>(0,2) = p_3d.at<float>(0,2) / p_3d.at<float>(0,3);
			p_3d.at<float>(0,3) = p_3d.at<float>(0,3) / p_3d.at<float>(0,3);


            PointT p;
            p.x = p_3d.at<float>(0,0);
            p.y = p_3d.at<float>(1,0);
            p.z = p_3d.at<float>(2,0);
            p.b = color.data[vL * color.step + uL * color.channels()];
            p.g = color.data[vL * color.step + uL * color.channels() + 1];
            p.r = color.data[vL * color.step + uL * color.channels() + 2];

			#pragma omp critical
			points.push_back(p);
        }
    }
} //omp

    //Depth filter and statistical removal
	cout << "inserting into cloud"<< endl;
	cloud->insert(cloud->begin(), points.begin(), points.end());

    PointCloud::Ptr updatedC(new PointCloud);
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(10);
    statistical_filter.setStddevMulThresh(0.1);
    statistical_filter.setInputCloud(cloud);
    statistical_filter.filter(*updatedC);

	cloud->clear();

    return updatedC;
}

