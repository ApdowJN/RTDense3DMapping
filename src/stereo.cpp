#include "stereo.h"
#include "imgio.hpp"
#include "sgm/StereoBMHelper.h"
#include "sgm/FastFilters.h"
#include "sgm/StereoSGM.h"
#include "post_helper.cpp"

void ncc(uint8* leftp, uint8 *rightp, double* cost, int* shape, int ndisp, int wsize){
    const int wc = wsize/2;
    const int sqwin = wsize*wsize;
    const int intgrrows = shape[0]+1;
    const int intgrcols = shape[1] +1;

    unsigned int * lintegral = (unsigned int *)calloc(intgrrows*intgrcols,sizeof(unsigned int));
    unsigned int * rintegral = (unsigned int *)calloc(intgrrows*intgrcols,sizeof(unsigned int));
    unsigned long long * sqlintegral = (unsigned long long *)calloc(intgrrows*intgrcols,sizeof(unsigned long long));
    unsigned long long * sqrintegral = (unsigned long long *)calloc(intgrrows*intgrcols,sizeof(unsigned long long));

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<shape[0]; i++){
			const int row = i*shape[1];
			const int introw = (i+1)*intgrcols;
			for (int j=0; j<shape[1]; j++){
				lintegral[introw+j+1] = leftp[row+j];
				rintegral[introw+j+1] = rightp[row+j];
			}
		}
    }

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<shape[0]; i++){
			const int row = i*shape[1];
			const int introw = (i+1)*intgrcols;
			for (int j=0; j<shape[1]; j++){
				sqlintegral[introw+j+1] = leftp[row+j]*leftp[row+j];
				sqrintegral[introw+j+1] =  rightp[row+j]* rightp[row+j];
			}
		}
    }	 

#pragma omp parallel num_threads(12)
    {
		for (int i=1; i< intgrrows; i++){
			const int row = i*intgrcols;
			const int prev_row = (i-1)*intgrcols;
			#pragma omp for
			for(int j=0; j<intgrcols;j++){
				lintegral[row+j] += lintegral[prev_row+j];
				rintegral[row+j] += rintegral[prev_row+j];
			}
		}
    }

#pragma omp parallel num_threads(12)
    {
		for (int i=1; i< intgrrows; i++){
			const int row = i*intgrcols;
			const int prev_row = (i-1)*intgrcols;
			#pragma omp for
			for(int j=0; j<intgrcols;j++){
				sqlintegral[row+j] += sqlintegral[prev_row+j];
				sqrintegral[row+j] += sqrintegral[prev_row+j];
			}
		}
    }    

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for(int i=0; i<intgrrows; i++){
			const int row =  i*intgrcols;
			for(int j=1; j<intgrcols; j++){
				lintegral[row+j] += lintegral[row+j-1];
				rintegral[row+j] += rintegral[row+j-1];
				sqlintegral[row+j] += sqlintegral[row+j-1];
				sqrintegral[row+j] += sqrintegral[row+j-1];
			}
		}
    }

    uint64* Al = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
    uint64* Ar = (uint64 *)calloc(shape[0]*shape[1],sizeof(uint64));
 
    double* Cl = (double *)calloc(shape[0]*shape[1],sizeof(double));
    double* Cr = (double *)calloc(shape[0]*shape[1],sizeof(double));

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for (int i=0; i< shape[0]-wsize;i++){
			const int row = (i+wc)*shape[1];
			const int t_row = i*intgrcols;
			const int b_row = (i+wsize)*intgrcols;
			for(int j=0; j< shape[1]-wsize; j++){
				const int col = j+wc;
				Al[row+col] = lintegral[b_row + j+wsize]+ lintegral[t_row + j]  - lintegral[b_row + j] - lintegral[t_row + j+wsize];
				Ar[row+col] = rintegral[b_row + j+wsize]+ rintegral[t_row + j] 	- rintegral[b_row + j] - rintegral[t_row + j+wsize];

			}
		}
    }

#pragma omp parallel num_threads(12)
    {
		#pragma omp for
		for (int i=0; i< shape[0]-wsize;i++){
			const int row = (i+wc)*shape[1];
			const int t_row = i*intgrcols;
			const int b_row = (i+wsize)*intgrcols;
			int j=0;
			#ifdef USE_AVX2
			for(; j< shape[1]-wsize-4; j+=4){
				const int col = j+wc;
  
				__m256i ymm1 = _mm256_set_epi64x( sqlintegral[b_row + j+wsize], sqlintegral[b_row + j+wsize+1],sqlintegral[b_row + j+wsize+2],sqlintegral[b_row + j+wsize+3]  );
				__m256i ymm2 = _mm256_set_epi64x( sqlintegral[b_row + j], sqlintegral[b_row + j+1], sqlintegral[b_row + j+2],sqlintegral[b_row + j+3] );
				__m256i ymm3 = _mm256_sub_epi64(ymm1,ymm2);
				ymm2 = _mm256_set_epi64x( sqlintegral[t_row + j], sqlintegral[t_row + j+1], sqlintegral[t_row + j+2],sqlintegral[t_row + j+3] );
				ymm3 = _mm256_add_epi64(ymm3,ymm2);
				ymm1 = _mm256_set_epi64x( sqlintegral[t_row + j+wsize], sqlintegral[t_row + j+wsize+1], sqlintegral[t_row + j+wsize+2],sqlintegral[t_row + j+wsize+3] );
				ymm3 = _mm256_sub_epi64(ymm3,ymm1);

				__m256i ymm4 = _mm256_set_epi64x( sqrintegral[b_row + j+wsize], sqrintegral[b_row + j+wsize+1],sqrintegral[b_row + j+wsize+2],sqrintegral[b_row + j+wsize+3]  );
				__m256i ymm5 = _mm256_set_epi64x( sqrintegral[b_row + j],sqrintegral[b_row + j+1], sqrintegral[b_row + j+2],sqrintegral[b_row + j+3] );
				__m256i ymm6 = _mm256_sub_epi64(ymm4,ymm5);
				ymm5 = _mm256_set_epi64x( sqrintegral[t_row + j], sqrintegral[t_row + j+1], sqrintegral[t_row + j+2],sqrintegral[t_row + j+3] );
				ymm6 = _mm256_add_epi64(ymm6,ymm5);
				ymm4 = _mm256_set_epi64x( sqrintegral[t_row + j+wsize], sqrintegral[t_row + j+wsize+1], sqrintegral[t_row + j+wsize+2],sqrintegral[t_row + j+wsize+3] );
				ymm6 = _mm256_sub_epi64(ymm6,ymm4);
				 							

				ymm1 = _mm256_set_epi64x( Al[row+col], Al[row+col+1],Al[row+col+2],Al[row+col+3]  );
				ymm2 = _mm256_mul_epi32( ymm1,ymm1 );

				__m256d ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm3,0),(double)_mm256_extract_epi64(ymm3,1), (double)_mm256_extract_epi64(ymm3,2),(double)_mm256_extract_epi64(ymm3,3) );
				__m256d ymm2d = _mm256_set1_pd((double) sqwin );
				__m256d ymm3d = _mm256_mul_pd (ymm1d, ymm2d);

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm2,0),(double)_mm256_extract_epi64(ymm2,1), (double)_mm256_extract_epi64(ymm2,2),(double)_mm256_extract_epi64(ymm2,3) );
				ymm3d = _mm256_sub_pd (ymm3d, ymm1d);
				ymm3d = _mm256_sqrt_pd (ymm3d);
				ymm2d = _mm256_set1_pd((double) 1 );
				ymm3d = _mm256_div_pd(ymm2d,ymm3d);

				_mm256_storeu_pd(&Cl[row+col],ymm3d);

				if(!std::isfinite(Cl[ row+col]))
					Cl[ row+col ] = 0;
				if(!std::isfinite(Cl[ row+col+1]))
					Cl[ row+col+1 ] = 0;				
				if(!std::isfinite(Cl[ row+col+2]))
					Cl[ row+col+2 ] = 0;				
				if(!std::isfinite(Cl[ row+col+3]))
					Cl[ row+col+3 ] = 0;		


				ymm1 = _mm256_set_epi64x( Ar[row+col], Ar[row+col+1],Ar[row+col+2],Ar[row+col+3]  );
				ymm2 = _mm256_mul_epi32( ymm1,ymm1 );

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm6,0),(double)_mm256_extract_epi64(ymm6,1), (double)_mm256_extract_epi64(ymm6,2),(double)_mm256_extract_epi64(ymm6,3) );
				ymm2d = _mm256_set1_pd((double) sqwin );
				ymm3d = _mm256_mul_pd (ymm1d, ymm2d);

				ymm1d = _mm256_set_pd( (double)_mm256_extract_epi64(ymm2,0),(double)_mm256_extract_epi64(ymm2,1), (double)_mm256_extract_epi64(ymm2,2),(double)_mm256_extract_epi64(ymm2,3) );
				ymm3d = _mm256_sub_pd (ymm3d, ymm1d);
				ymm3d = _mm256_sqrt_pd (ymm3d);
				ymm2d = _mm256_set1_pd((double) 1 );
				ymm3d = _mm256_div_pd(ymm2d,ymm3d);

				_mm256_storeu_pd(&Cr[row+col],ymm3d);							

				if(!std::isfinite(Cr[ row+col]))
					Cr[ row+col ] = 0;
				if(!std::isfinite(Cr[ row+col+1]))
					Cr[ row+col+1 ] = 0;				
				if(!std::isfinite(Cr[ row+col+2]))
					Cr[ row+col+2 ] = 0;				
				if(!std::isfinite(Cr[ row+col+3]))
					Cr[ row+col+3 ] = 0;							
			}
			#endif

			for(; j< shape[1]-wsize; j++){
				const int col = j+wc;

				unsigned long long Bl = sqlintegral[b_row + j+wsize] + sqlintegral[t_row + j] - sqlintegral[b_row + j] - sqlintegral[t_row + j+wsize];
				unsigned long long Br = sqrintegral[b_row + j+wsize] + sqrintegral[t_row + j] - sqrintegral[b_row + j] - sqrintegral[t_row + j+wsize];

				Cl[ row+col ] = 1/(sqrt(sqwin*Bl - (double)( Al[row+col] )*( Al[row+col] ) ));
				if(!std::isfinite(Cl[ row+col]))
					Cl[ row+col ] = 0;

				Cr[ row+col ] = 1/(sqrt(sqwin*Br - (double)( Ar[row+col] )*( Ar[row+col]) ));
				if(!std::isfinite(Cr[ row+col]))
					Cr[ row+col ] = 0;				
			}			
		}
    }   
      	
#pragma omp parallel num_threads(12)
    {
		double * dslice = (double*)calloc(intgrrows*intgrcols,sizeof(double));
		#pragma omp for
		for (int d=0; d<ndisp; d++ ){
			const int d_row = d*shape[0]*shape[1];
			std::fill_n(dslice,intgrrows*intgrcols,0);
			for(int i=0; i<shape[0]; i++){
				const int row = i*shape[1];
				const int intgrrow = (i+1)*intgrcols;
				for(int j=d; j<shape[1]; j++){
					dslice[intgrrow + j+1] = leftp[row+j]*rightp[row+j-d];
				}
			}

			for(int i=1; i<intgrrows; i++ ){
				const int row = i*intgrcols;
				const int prev_row = (i-1)*intgrcols;
				for(int j=0; j<intgrcols; j++){
					dslice[row + j] += dslice[prev_row + j];
				}

			}
 
		int iu=0;
		for( ; iu<intgrrows-8; iu+=8 ){
			const int rowind = iu*intgrcols;
			const int rowind1 = (iu+1)*intgrcols;
			const int rowind2 = (iu+2)*intgrcols;
			const int rowind3 = (iu+3)*intgrcols;
			const int rowind4 = (iu+4)*intgrcols;
			const int rowind5 = (iu+5)*intgrcols;
			const int rowind6 = (iu+6)*intgrcols;
			const int rowind7 = (iu+7)*intgrcols;			
			for(int j=d+1; j<intgrcols; j++){
				double s0, s1;
				s0 = dslice[rowind+j-1];
				s1 = dslice[rowind+j];
				dslice[rowind+j] = s1+s0;
				
				s0 = dslice[rowind1+j-1];
				s1 = dslice[rowind1+j];
				dslice[rowind1+j] = s1+s0;

				s0 = dslice[rowind2+j-1];
				s1 = dslice[rowind2+j];
				dslice[rowind2+j] = s1+s0;

				s0 = dslice[rowind3+j-1];
				s1 = dslice[rowind3+j];
				dslice[rowind3+j] = s1+s0;


				s0 = dslice[rowind4+j-1];
				s1 = dslice[rowind4+j];
				dslice[rowind4+j] = s1+s0;				

				s0 = dslice[rowind5+j-1];
				s1 = dslice[rowind5+j];
				dslice[rowind5+j] = s1+s0;

				s0 = dslice[rowind6+j-1];
				s1 = dslice[rowind6+j];
				dslice[rowind6+j] = s1+s0;			

				s0 = dslice[rowind7+j-1];
				s1 = dslice[rowind7+j];
				dslice[rowind7+j] = s1+s0;					
			}
		}

		for( ; iu<intgrrows; iu++){
			const int rowind = iu*intgrcols;
			for(int j=d+1; j<intgrcols; j++){
				dslice[rowind+j] += dslice[rowind+j-1];
			}
		}

			for (int i=0; i< shape[0]-wsize; i++){
				const int row = (i+wc)*shape[1];
				const int t_row = i*intgrcols;
				const int b_row = (i+wsize)*intgrcols;
				int j=d;
				#ifdef USE_AVX2 
				for(; j< shape[1]-wsize-4; j+=4){
					const int col = (j+wc);
					__m256d ymm1 = _mm256_loadu_pd (&dslice[b_row + j+wsize ]);
					__m256d ymm2 = _mm256_loadu_pd (&dslice[b_row +j ]);
					__m256d ymm3 = _mm256_sub_pd (ymm1, ymm2);
					ymm2 = _mm256_loadu_pd (&dslice[t_row+j]);
					ymm3 = _mm256_add_pd (ymm3, ymm2);
					ymm1 = _mm256_loadu_pd (&dslice[t_row +j+wsize ]);
					ymm3 = _mm256_sub_pd (ymm3, ymm1);
					ymm1 = _mm256_set1_pd((double) sqwin );
					ymm3 = _mm256_mul_pd (ymm3, ymm1);

					__m256i ymm4 = _mm256_set_epi64x( Al[row+col], Al[row+col+1], Al[row+col+2],Al[row+col+3] );
					__m256i ymm5 = _mm256_set_epi64x( Ar[row+(j-d+wc)], Ar[row+(j-d+wc)+1], Ar[row+(j-d+wc)+2],Ar[row+(j-d+wc)+3] );

					__m256i ymm6 = _mm256_mul_epi32( ymm4,ymm5 );

					ymm1 = _mm256_set_pd( (double)_mm256_extract_epi64(ymm6,0),(double)_mm256_extract_epi64(ymm6,1), (double)_mm256_extract_epi64(ymm6,2),(double)_mm256_extract_epi64(ymm6,3) );

					ymm3 = _mm256_sub_pd(ymm3,ymm1);

					ymm1 = _mm256_loadu_pd (&Cl[ row+col ]);
					ymm2 = _mm256_loadu_pd (&Cr[ row+(j-d+wc) ]);
					ymm2 = _mm256_mul_pd(ymm1,ymm2);
					ymm3 = _mm256_mul_pd(ymm3,ymm2);
					ymm1 = _mm256_set1_pd((double) -1 );
					ymm3 = _mm256_mul_pd(ymm3,ymm1);

					_mm256_storeu_pd(&cost[d_row + row+col],ymm3);
				}
#endif

				for(; j< shape[1]-wsize; j++){
					const int col = (j+wc);
					const double lD = dslice[b_row + j+wsize ] + dslice[t_row+j]
								 	- dslice[b_row +j ] - dslice[t_row +j+wsize ];
			        cost[d_row + row+col] = -(sqwin*lD- Al[row+col] * Ar[row+(j-d+wc)]) *Cl[ row+col ]*Cr[ row+(j-d+wc) ];
				}				
			}
		}

		delete [] dslice;
    }

    delete [] lintegral;
    delete [] rintegral;
    delete [] sqlintegral;
    delete [] sqrintegral;
    delete [] Al;
    delete [] Ar;
    delete [] Cl;
    delete [] Cr;
}

void sad(uint8* leftp, uint8 *rightp, uint *cost, int* shape, int ndisp, int wsize){

    const int integrrows = shape[0]+1;
    const int integrcols = shape[1]+1;


#pragma omp parallel num_threads(12)
    {
        uint * slice = new uint[integrrows*integrcols];

        const int wc = wsize/2;
#pragma omp for
        for (int d=0; d<ndisp; d++ ){

            const int dind = d*shape[0]*shape[1];
            uint* res_data = cost+dind;
            std::fill_n(slice,integrrows*integrcols,0);

            for( int i=0; i<shape[0]; i++){
                const int rowind = i*shape[1];
                const int intgrrow = (i+1)*integrcols+1;
                for(int j=d; j<shape[1]; j++){
                    slice[intgrrow+j] = abs( leftp[rowind+j] - rightp[rowind+(j-d)] );
                }
            }


            for( int i=1; i<integrrows; i++ ){

                const int prev_row = (i-1)*integrcols;
                const int intgrrow = i*integrcols;
                for(int j=d; j<integrcols; j++){
                    slice[intgrrow+j] += slice[prev_row+j];
                }
            }


            int iu=0;
            for( ; iu<integrrows-8; iu+=8 ){
                const int rowind = iu*integrcols;
                const int rowind1 = (iu+1)*integrcols;
                const int rowind2 = (iu+2)*integrcols;
                const int rowind3 = (iu+3)*integrcols;
                const int rowind4 = (iu+4)*integrcols;
                const int rowind5 = (iu+5)*integrcols;
                const int rowind6 = (iu+6)*integrcols;
                const int rowind7 = (iu+7)*integrcols;
                for(int j=d+1; j<integrcols; j++){

                    int s0; int s1;
                    s0 = slice[rowind+j-1];
                    s1 = slice[rowind+j];
                    slice[rowind+j] = s1+s0;

                    s0 = slice[rowind1+j-1];
                    s1 = slice[rowind1+j];
                    slice[rowind1+j] = s1+s0;

                    s0 = slice[rowind2+j-1];
                    s1 = slice[rowind2+j];
                    slice[rowind2+j] = s1+s0;

                    s0 = slice[rowind3+j-1];
                    s1 = slice[rowind3+j];
                    slice[rowind3+j] = s1+s0;


                    s0 = slice[rowind4+j-1];
                    s1 = slice[rowind4+j];
                    slice[rowind4+j] = s1+s0;

                    s0 = slice[rowind5+j-1];
                    s1 = slice[rowind5+j];
                    slice[rowind5+j] = s1+s0;

                    s0 = slice[rowind6+j-1];
                    s1 = slice[rowind6+j];
                    slice[rowind6+j] = s1+s0;

                    s0 = slice[rowind7+j-1];
                    s1 = slice[rowind7+j];
                    slice[rowind7+j] = s1+s0;

                }
            }

            for( ; iu<integrrows; iu++){
                const int rowind = iu*integrcols;
                for(int j=d+1; j<integrcols; j++){
                    slice[rowind+j] += slice[rowind+j-1];
                }
            }


            for(int i=0; i<shape[0]-wsize;i++){
                const int place_row =(i+wc)*shape[1]+wc;
                const int t_row = i*integrcols;
                const int b_row = (i+wsize)*integrcols;

                for(int j=d; j<shape[1]-wsize; j++){
                    res_data[place_row+j] = slice[b_row+(j+wsize)  ] - slice[b_row+j ] + slice[t_row+j]   - slice[t_row+(j+wsize) ] ;
                }

            }


        }

        delete []  slice;

    }

}

void colorMat2Array(const cv::Mat &imColor) {
    int height = imColor.rows;
    int width = imColor.cols;

    uint8_t *pRGB = new uint8_t[height*width*3];
    for (int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            for(uint k = 0; k < 3; ++k) {
                pRGB[i * width * 3 + j * 3 + k] = imColor.at<cv::Vec3b>(i, j)[k];
            }
        }
    }
}

void grayMat2Array(const cv::Mat &imGray, uint8_t* pGray)  {
    int height = imGray.rows;
    int width = imGray.cols;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                pGray[i * width + j] = imGray.at<uchar>(i, j);
            }
        }
    }
}

void runStereo(int ndisp, int wsize, bool post, string method, const string &filename, const cv::Mat &imLeftGray, const cv::Mat &imRightGray, cv::Mat &disp, cv::Mat &conf, cv::Mat &cost1, cv::Mat &cost2, cv::Mat &cost3) {
    int numStrips = 12;
    StereoSGMParams_t params;
    params.InvalidDispCost=63;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = 8;
    params.subPixelRefine = 1;
    params.NoPasses = 2;
    params.rlCheck = false;

    float scale;

    if(post){
		if( params.lrCheck ){
			if(ndisp%16 != 0){
				ndisp=ndisp+(16 - ndisp%16);
			}
		}else{
			if(ndisp%8 != 0)
				ndisp=ndisp+(8 - ndisp%8);
		}
	}

    int imHeight = imLeftGray.rows;
    int imWidth  = imRightGray.cols;

    int* shape = new int[3];

    int width = imWidth;
    shape[0]=imHeight;shape[2]=ndisp;
    if(post){
        width= imWidth+(16-imWidth%16);
    }
    shape[1]=width;

    uint16* cost_vect1=(uint16*)calloc(width*imHeight,sizeof(uint16));
    std::fill_n(cost_vect1,width*imHeight, 0);
    uint16* cost_vect2=(uint16*)calloc(width*imHeight,sizeof(uint16));
    std::fill_n(cost_vect2,width*imHeight, 0);
    uint16* cost_vect3=(uint16*)calloc(width*imHeight,sizeof(uint16));
    std::fill_n(cost_vect3,width*imHeight, 0);

    float* conf_vect = (float*)_mm_malloc(width*imHeight*ndisp*sizeof(float), 16);
    float* disp_vect = (float*)_mm_malloc(width*imHeight*sizeof(float), 16);

    uint8* imgl= (uint8*)calloc(imHeight * imWidth, sizeof(uint8));
    uint8* imgr= (uint8*)calloc(imHeight * imWidth, sizeof(uint8));

    grayMat2Array(imLeftGray, imgl);
    grayMat2Array(imRightGray, imgr);


    if(post){
        imgl = paddwidth( imgl, imHeight, imWidth);
        imgr = paddwidth( imgr, imHeight, imWidth);
    }

	if (method=="sad"){
    	uint* cost_vect=(uint*)calloc(width*imHeight*ndisp,sizeof(uint));

		scale = (float)63/(255*wsize);
    	std::fill_n(cost_vect,shape[0]*shape[1]*ndisp, wsize*wsize*255);
    	std::fill_n(conf_vect,shape[0]*shape[1]*ndisp, 0.0);

    	sad(imgl, imgr, cost_vect, shape, ndisp, wsize);
		disp_vect = doPost(disp_vect, cost_vect, conf_vect, cost_vect1, cost_vect2, cost_vect3, shape ,imgl, scale, 0, numStrips, params);
    	free(cost_vect);

	} else if(method == "ncc") {
		double* cost_vect=(double*)calloc(width*imHeight*ndisp,sizeof(double));

		scale = 32;
    	std::fill_n(cost_vect,shape[0]*shape[1]*ndisp, wsize*wsize-1);
    	std::fill_n(conf_vect,shape[0]*shape[1]*ndisp, 0.0);

    	ncc(imgl, imgr, cost_vect, shape, ndisp, wsize);
		disp_vect = doPost(disp_vect, cost_vect, conf_vect, cost_vect1, cost_vect2, cost_vect3, shape ,imgl, scale, 1, numStrips, params);
    	free(cost_vect);

	} else {
		cout << "Error: " << method << " not yet implemented." << endl;
		exit(EXIT_FAILURE);
	}

	// build disp, cost, and conf Mats
	disp = cv::Mat(shape[0], shape[1], CV_32F, disp_vect);
	conf = cv::Mat(shape[0], shape[1], CV_32F, conf_vect);
	cost1 = cv::Mat(shape[0], shape[1], CV_16UC1, cost_vect1);
	cost2 = cv::Mat(shape[0], shape[1], CV_16UC1, cost_vect2);
	cost3 = cv::Mat(shape[0], shape[1], CV_16UC1, cost_vect3);

    free(imgl);
    free(imgr);
    delete [] shape;
}
