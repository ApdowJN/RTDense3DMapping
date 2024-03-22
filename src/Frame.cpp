//
// Created by wangweihan on 10/17/21.
//

#include "Frame.h"
Frame::Frame(){}

//Copy Constructor
Frame::Frame(const Frame &frame) :
	mBaseline(frame.mBaseline),
	mvuRight(frame.mvuRight),
	mIndex(frame.mIndex),
	mSize(frame.mSize),
	mK(frame.mK),
	mDepthMap(frame.mDepthMap.clone()),
	mConfMap(frame.mConfMap.clone()),
	mTcw(frame.mTcw.clone()),
	mColorMap(frame.mColorMap.clone())
{}

Frame::Frame(cv::Mat &K, const float &baseline, const long unsigned int &sequenceId, const cv::Size &imgSize) :
	mK(K),
	mBaseline(baseline),
	mIndex(sequenceId),
	mSize(imgSize)
{}

void Frame::setPose(cv::Mat Tcw_) {
    Tcw_.copyTo(mTcw);
//    mRcw = mTcw.rowRange(0,3).colRange(0,3);
//    mtcw = mTcw.rowRange(0,3).col(3);
}

void Frame::setColorMap(cv::Mat imColor_)
{
    imColor_.copyTo(mColorMap);
}

void Frame::getDepthMap(const cv::Mat &imDisparity) {
    int rows = mSize.height;
    int cols = mSize.width;
    int count = 0;
    int offset = 0;
    cv::Mat res = cv::Mat::zeros(mSize, CV_32F);
	const float fx = mK.at<float>(0,0);

#pragma omp parallel num_threads(12)
    {
#pragma omp for
        for (int vL = offset; vL < rows - offset; ++vL)
        {
            for (int uL = offset; uL < cols - offset; ++uL)
            {
                float d = (float) imDisparity.at<float>(vL, uL);

				if (d <= 1.0) {
					continue;
				}

                float depth = (mBaseline*fx) / d;

				//if (depth <= 0.0 || depth >= 35*mBaseline) //tune this 20.20*mb
				//    continue;

                res.at<float>(vL, uL) = depth;
                ++count;
            }
        }
    }
    res.copyTo(mDepthMap);
}

void Frame::getDepthMap()
{
    int rows = mSize.height;
    int cols = mSize.width;
	const float fx = mK.at<float>(0,0);

    int offset = 0;
    int count = 0;
    cv::Mat res = cv::Mat::zeros(mSize, CV_32F);

#pragma omp parallel num_threads(12)
{
#pragma omp for
    for (int vL = 0; vL < rows - offset; ++vL) {
        for (int uL = offset; uL < cols - offset; ++uL) {
            float uR = mvuRight[vL * cols + uL];

            if (uR == -1)
                continue;

            float depth = fx * (mBaseline / (uL - uR));

            res.at<float>(vL, uL) = depth;
        }
    }
} //omp
	

    res.copyTo(mDepthMap);
}

void Frame::load_depth_map(const cv::Size &imageSize, const cv::Mat &dispMat, const float &baseline)
{
    const int rows = imageSize.height;
    const int cols = imageSize.width;

    cv::Mat depthi = cv::Mat::zeros(imageSize, CV_32F);
    //#pragma omp parallel
    // {
    //#pragma omp for
    for(int v = 0; v < rows; v++)
    {
        for(int u = 0; u < cols; u++)
        {
            size_t d = dispMat.at<ushort>(v, u);

            if(d == 0)
                continue;

            d /=256;

            float depth = mBaseline/d;

            if(depth <= 0.1 || depth >=35*baseline)
                continue;

            depthi.at<float>(v, u) = depth;
        }
    }
    //}
    depthi.copyTo(mDepthMap);


}

void Frame::load_confidence_map(const cv::Size &imageSize, const string &filePath)
{
    const int rows = imageSize.height;
    const int cols = imageSize.width+16;

    //remove ".png"
    string token = filePath;
    if(cols == 816)
    {
        int len = filePath.size();
        len -= 4;
        token = filePath.substr(0, len);
    }

    string filename = token + ".pfm";

    FILE * pFile;
    float* buffer = (float*)calloc(rows*cols,sizeof(float));;
    pFile = fopen(filename.c_str(),"rb");
    char c[100];
    int width, height, endianess;
    if (pFile != NULL)
    {
        int res = fscanf(pFile, "%s", c);
        if(res !=EOF && !strcmp(c,"Pf"))
        {
            res = fscanf(pFile, "%s", c);
            width = atoi(c);
            res =fscanf(pFile, "%s", c);
            height = atoi(c);
            res = fscanf(pFile, "%s", c);
            endianess = atof(c);

            fseek (pFile , 0, SEEK_END);
            long lSize = ftell (pFile);
            long pos = lSize - width*height*4;
            fseek (pFile , pos, SEEK_SET);
            float* img = new float[width*height];
            size_t result =fread(img,sizeof(float),width*height,pFile);

            //PFM SPEC image stored bottom -> top reversing image
            if(result >0)
            {
#pragma omp parallel
                {
#pragma omp for
                    for(int i = 0; i< height; i++)
                    {
                        memcpy(&buffer[(height-i-1)*width],&img[(i*width)],width*sizeof(float));
                    }
                }
            }
            delete[] img;
        }
        else
        {
            std::cout << "Invalid magic number! " <<std::endl;
            fclose(pFile);
            exit(1);
        }

    }
    else
    {
        std::cout << "File " << filename << " does not exist!" << std::endl;
        fclose(pFile);
        exit(1);
    }
    fclose(pFile);
//        Mat confi = Mat(height, width, CV_32FC1);
    cv::Mat confi = cv::Mat(rows, cols, CV_32F);
#pragma omp parallel
    {
#pragma omp for
        for(uint i=0;i<rows;i++)
        {
            for(uint j=0;j<cols;j++)
            {
                confi.at<float>(i, j) = buffer[i*width+j];

            }
        }
    }
    mConfMap = confi.clone();
    free(buffer);
}

void Frame::parabola_fitting(const cv::Mat &cost1, const cv::Mat &cost2, const cv::Mat &cost3, const cv::Mat &imDisparity) {
    const int rows = mSize.height;
    const int cols = mSize.width;
	const float fx = mK.at<float>(0,0);

    //const float minZ = mBaseline;
    const float minD = 0;            // minimum disparity
    const float maxD = fx;    // maximum disparity
    mvuRight.resize(rows * cols, -1);

//    int count = 0;
    auto begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(12)
    {
        std::vector<std::pair<int, int> > vDistIdx;
#pragma omp for
        for (int vL = 0; vL < rows; ++vL) {
            for (int uL = 0; uL < cols; ++uL) {
                // search range, this range theoretical
                const float minU = uL - maxD;
                const float maxU = uL - minD;

                // do not have match point
                if (maxU < 0)
                    continue;

                float disparity = (float) imDisparity.at<float>(vL, uL);

                if (disparity <= 10.0 || disparity >= 100)
                    continue;

                float uR = uL - disparity;

                // if over[minU, maxU]，this is mismatched, drop it
                if (uR >= minU && uR <= maxU) {
                    const float dist1 = (float)cost1.at<ushort>(vL, uL);
                    const float dist2 = (float)cost2.at<ushort>(vL, uL);
                    const float dist3 = (float)cost3.at<ushort>(vL, uL);

                    if(dist1 + dist3 == 2.0*dist2)
                        continue;

                    const float deltaDisparity = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

                    if (deltaDisparity < -0.5 || deltaDisparity > 0.5)
                        continue;

                    float bestuR = uR + deltaDisparity;

                    mvuRight[(vL * cols) + uL] = bestuR;

                }
            }
        }
    } // omp

	// produce depth map
    getDepthMap();
}


