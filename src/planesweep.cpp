//
// Created by wangweihan on 12/3/22.
//
#include "planesweep.h"

PlaneSweep::PlaneSweep(){}

PlaneSweep::PlaneSweep(const int &width, const int &height):mwidth(width), mheight(height){
    depth_map = cv::Mat::zeros(mheight, mwidth, CV_32F);
}

void PlaneSweep::configPlanes(float min_depth, float max_depth, int num_planes){
    plane_depths.resize(num_planes+1);
    this->depth_interval = (max_depth-min_depth)/num_planes;

    plane_depths[0] = min_depth;
    plane_depths[ num_planes] = max_depth;

    for (int i=1; i<num_planes; i++){
        plane_depths[i] = min_depth+i*depth_interval;
    }

}

//H = Hcr
double * PlaneSweep::Homography(const cv::Mat &K_ref_inv, const cv::Mat &Rcr, const cv::Mat &tcr, const cv::Mat &K_ref, std::vector<double> normal, double d){
    double * H = (double*)calloc(9,sizeof(double));
    double * trans = (double*)calloc(9,sizeof(double));
    trans[0] = tcr.at<float>(0)*normal[0]/d;
    trans[1] = tcr.at<float>(0)*normal[1]/d;
    trans[2] = tcr.at<float>(0)*normal[2]/d;
    trans[3] = tcr.at<float>(1)*normal[0]/d;
    trans[4] = tcr.at<float>(1)*normal[1]/d;
    trans[5] = tcr.at<float>(1)*normal[2]/d;
    trans[6] = tcr.at<float>(2)*normal[0]/d;
    trans[7] = tcr.at<float>(2)*normal[1]/d;
    trans[8] = tcr.at<float>(2)*normal[2]/d;

    double* ext= (double*)calloc(9,sizeof(double));
    ext[0] = Rcr.at<float>(0,0)+trans[0]; ext[1] = Rcr.at<float>(0,1)+trans[1]; ext[2] = Rcr.at<float>(0,2)+trans[2];
    ext[3] = Rcr.at<float>(1,0)+trans[3]; ext[4] = Rcr.at<float>(1,1)+trans[4]; ext[5] = Rcr.at<float>(1,2)+trans[5];
    ext[6] = Rcr.at<float>(2,0)+trans[6]; ext[7] = Rcr.at<float>(2,1)+trans[7]; ext[8] = Rcr.at<float>(2,2)+trans[8];


    double* temp= (double*)calloc(9,sizeof(double));
    temp[0] = ext[0]*K_ref_inv.at<float>(0,0) + ext[1]*K_ref_inv.at<float>(1,0) + ext[2]*K_ref_inv.at<float>(2,0);
    temp[1] = ext[0]*K_ref_inv.at<float>(0,1) + ext[1]*K_ref_inv.at<float>(1,1) + ext[2]*K_ref_inv.at<float>(2,1);
    temp[2] = ext[0]*K_ref_inv.at<float>(0,2) + ext[1]*K_ref_inv.at<float>(1,2) + ext[2]*K_ref_inv.at<float>(2,2);

    temp[3] = ext[3]*K_ref_inv.at<float>(0,0) + ext[4]*K_ref_inv.at<float>(1,0) + ext[5]*K_ref_inv.at<float>(2,0);
    temp[4] = ext[3]*K_ref_inv.at<float>(0,1) + ext[4]*K_ref_inv.at<float>(1,1) + ext[5]*K_ref_inv.at<float>(2,1);
    temp[5] = ext[3]*K_ref_inv.at<float>(0,2) + ext[4]*K_ref_inv.at<float>(1,2) + ext[5]*K_ref_inv.at<float>(2,2);

    temp[6] = ext[6]*K_ref_inv.at<float>(0,0) + ext[7]*K_ref_inv.at<float>(1,0) + ext[8]*K_ref_inv.at<float>(2,0);
    temp[7] = ext[6]*K_ref_inv.at<float>(0,1) + ext[7]*K_ref_inv.at<float>(1,1) + ext[8]*K_ref_inv.at<float>(2,1);
    temp[8] = ext[6]*K_ref_inv.at<float>(0,2) + ext[7]*K_ref_inv.at<float>(1,2) + ext[8]*K_ref_inv.at<float>(2,2);


    H[0] = K_ref.at<float>(0,0)*temp[0] + K_ref.at<float>(0,1)*temp[3] + K_ref.at<float>(0,2)*temp[6];
    H[1] = K_ref.at<float>(0,0)*temp[1] + K_ref.at<float>(0,1)*temp[4] + K_ref.at<float>(0,2)*temp[7];
    H[2] = K_ref.at<float>(0,0)*temp[2] + K_ref.at<float>(0,1)*temp[5] + K_ref.at<float>(0,2)*temp[8];

    H[3] = K_ref.at<float>(1,0)*temp[0] + K_ref.at<float>(1,1)*temp[3] + K_ref.at<float>(1,2)*temp[6];
    H[4] = K_ref.at<float>(1,0)*temp[1] + K_ref.at<float>(1,1)*temp[4] + K_ref.at<float>(1,2)*temp[7];
    H[5] = K_ref.at<float>(1,0)*temp[2] + K_ref.at<float>(1,1)*temp[5] + K_ref.at<float>(1,2)*temp[8];

    H[6] = K_ref.at<float>(2,0)*temp[0] + K_ref.at<float>(2,1)*temp[3] + K_ref.at<float>(2,2)*temp[6];
    H[7] = K_ref.at<float>(2,0)*temp[1] + K_ref.at<float>(2,1)*temp[4] + K_ref.at<float>(2,2)*temp[7];
    H[8] = K_ref.at<float>(2,0)*temp[2] + K_ref.at<float>(2,1)*temp[5] + K_ref.at<float>(2,2)*temp[8];

    free(temp);
    free(trans);
    free(ext);

    return H;
}


inline uint8 PlaneSweep::get_val(const cv::Mat &img, double u, double v){
    if(floor(u)<0 || floor(v)< 0 || ceil(v) >= mheight || ceil(u) >= mwidth)
        return 0;

    int u_low = floor(u);
    int u_high = ceil(u);
    int v_low = floor(v);
    int v_high = ceil(v);
    double w = (double)u - u_low;
    double h = (double)v - v_low;

    double a = (img.at<uchar>(v_low, u_low)*(1-w)*(1-h));
    double b = (img.at<uchar>(v_low, u_high)*(1-h)*w);
    double c = (img.at<uchar>(v_high, u_low)*h*(1-w));
    double d = (img.at<uchar>(v_high, u_high)*w*h);
    return (uint8)a+b+c+d;
}

// pixels are out of bounds are set to zero.
void PlaneSweep::interpolate(const int &globalId, double* H, uint8 *im_h){
//#pragma omp parallel num_threads(12)
    //{
    //#pragma omp for collapse(2)
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    float duration;
    vector<float> get_val_timings;

    for(int v = 0; v < mheight; v++){
        for(int u = 0; u < mwidth; u++){

            double u2 = H[0]*u + H[1]*v +H[2];
            double v2 = H[3]*u + H[4]*v +H[5];
            double w = H[6]*u + H[7]*v +H[8];
            u2 = u2/w;
            v2 = v2/w;
            start = std::chrono::high_resolution_clock::now();
            im_h[v*mwidth +u] = get_val(imgs[globalId], u2, v2);
            end = std::chrono::high_resolution_clock::now();
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            get_val_timings.push_back(duration);
        }
    }
//    double sum = 0;
//    int n = get_val_timings.size();
//    double total_gv_t = std::accumulate(get_val_timings.begin(), get_val_timings.end(), sum); //ms
//    cout << "total timing of get_val: " << total_gv_t/1000 << " s,"<<" average time: "<<total_gv_t/(1000*n)<<" s." << endl;
    //}
}


void PlaneSweep::SAD( uint8* im_h, int w,int d){
    int w_c = w/2;
    double * vd = &volume[d*mheight*mwidth];
    for(int i = 0; i < mheight-w; i++){
        for(int j = 0; j < mwidth-w; j++){
            for(int wh=0; wh<w;wh++){
                for(int ww=0;ww<w;ww++){
                    vd[(i+w_c)*mwidth+(j+w_c)] += abs((int)imgs[ref_global_id].at<uchar>((i+wh),(j+ww)) - (int)im_h[(i+wh)*mwidth + (j+ww)]);
                }
            }
        }
    }
}

void PlaneSweep::WTA(){
    depth_map = cv::Mat::zeros(mheight, mwidth, CV_32F);
    for(int i=0; i<mheight; i++){
        for(int j=0; j<mwidth; j++){
            double c = volume[i*mwidth+j]; //d == 0
            for(size_t d = 1; d<plane_depths.size(); d++){
                double cc = volume[d*mheight*mwidth+ i*mwidth+j];
                if(c>cc){
                    c = cc;
                    depth_map.at<float>(i,j) = plane_depths[d];
                }
            }
        }
    }
}

void PlaneSweep::sweep(const deque<Frame> &keyframesCache, std::vector<double> normal){
    int num_views = keyframesCache.size();
    int localRefIndex = num_views / 2;
    int num_depths = plane_depths.size();


    volume = (double*)calloc(num_depths*mheight*mwidth,sizeof(double));
//    depth = (uint8*)calloc(mheight*mwidth,sizeof(uint8));

    Frame ReferenceFrame = keyframesCache.at(localRefIndex);
    cv::Mat Rrw = ReferenceFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat Rwr = Rrw.t();
    cv::Mat trw = ReferenceFrame.mTcw.rowRange(0,3).col(3);
    cv::Mat twr = -Rwr*trw;

    cv::Mat K = ReferenceFrame.mK;
    cv::Mat K_inv = K.inv();

    // timing
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    float duration = 0;
    ref_global_id = ReferenceFrame.mIndex;
    cout<<"Plane sweep for Frame "<<ref_global_id<<endl;
    vector<float> h_timings;
    vector<float> interpolate_timings;
    vector<float> sad_timings;
    for(int i = 0; i < num_depths; ++i){
        cout << "sweeping: " << i << endl;
        for(int f = -localRefIndex; f <= localRefIndex; ++f) {
            if (f == 0)
                continue;
            uint8* im_h = (uint8*)calloc(mheight*mwidth,sizeof(uint8));
            Frame neighbor = keyframesCache.at(localRefIndex + f);
            cv::Mat Rcw = neighbor.mTcw.rowRange(0,3).colRange(0,3);
            cv::Mat Rcr = Rcw*Rwr;

            cv::Mat tcw = neighbor.mTcw.rowRange(0,3).col(3);
            cv::Mat tcr = Rcw*twr + tcw;

            start = std::chrono::high_resolution_clock::now();
            double * H = Homography(K_inv, Rcr, tcr, K, normal, plane_depths[i]);
            end = std::chrono::high_resolution_clock::now();
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            h_timings.push_back(duration);


            start = std::chrono::high_resolution_clock::now();
            interpolate(ref_global_id+f, H, im_h);
            end = std::chrono::high_resolution_clock::now();
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            interpolate_timings.push_back(duration);

            // compute cost for each depth plane
            start = std::chrono::high_resolution_clock::now();
            SAD(im_h, 7, i);
            end = std::chrono::high_resolution_clock::now();
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())/1000;
            sad_timings.push_back(duration);

            free(H);
            free(im_h);
        }
    }

    int nh = h_timings.size(), ni = interpolate_timings.size(), ns = sad_timings.size();
    cout<<nh<<", "<<ni<<", "<<ns<<endl;
    double sum = 0;
    double total_H_t = std::accumulate(h_timings.begin(), h_timings.end(), sum); //ms
    cout << "total timing of Homography: " << total_H_t/1000 << " s,"<<" average time: "<<total_H_t/(1000*nh)<<" s." << endl;

    sum = 0;
    double total_I_t = std::accumulate(interpolate_timings.begin(), interpolate_timings.end(), sum);
    cout << "total timing of interpolate: " << total_I_t/1000 << " s,"<<" average time: "<<total_I_t/(1000*ni)<<" s." << endl;
    if (total_I_t > 60000) {
        cout<<total_I_t/60000<< "min."<<endl;
    }
    sum = 0;
    double total_S_t = std::accumulate(sad_timings.begin(), sad_timings.end(), sum);
    cout << "total timing of sad: " << total_S_t/1000 << " s,"<<" average time: "<<total_S_t/(1000*ns)<<" s." << endl;


    WTA();
}

PlaneSweep::~PlaneSweep(){
    free(volume);
}
