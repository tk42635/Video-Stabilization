//Call functions VideoStab::Estimate() and then VideoStab::Compensate() to stabilize one frame

#include "stablization.h"
#include <cmath>


//Parameters for KalmanFilter
#define Q1 0.004
#define R1 0.5

//sigma for GausianFilter
#define sigma 40

VideoStab::VideoStab()
{

    smoothedMat.create(2 , 3 , CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 1;
    scaleY = 1;
    thetha = 0;
    transX = 0;
    transY = 0;

}

//Motion Estimate
//Input: PreviousFrame and CurrentFrame in grayscale
//Output: Motion parameters(absolute value)
vector<double> VideoStab::Estimate(Mat frame1, Mat frame2)
{
    vector <Point2f> features1, features2;
    vector <Point2f> goodFeatures1, goodFeatures2;
    vector <Point2f> RS_Features1, RS_Features2;
    vector <uchar> status;
    vector <float> err;


    goodFeaturesToTrack(frame1, features1, 300, 0.01  , 30 );
     
    if(features1.size() <= 20)
        return tmp;
        
    calcOpticalFlowPyrLK(frame1, frame2, features1, features2, status, err);
    if(features2.size() <= 20)
        return tmp;

    for(size_t i=0; i < status.size(); i++)
    {
        if(status[i])
        {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);
        }
    }
    if(goodFeatures1.size() <= 20 || goodFeatures2.size() <= 20)
        return tmp;

    //RANSAC
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(goodFeatures1, goodFeatures2, RansacStatus, FM_RANSAC);
    for(size_t i=0; i < status.size(); i++)
    {
        if(RansacStatus[i] != 0)
        {
            RS_Features1.push_back(features1[i]);
            RS_Features2.push_back(features2[i]);
        }
    }
    if(RS_Features1.size() <= 20 || RS_Features2.size() <= 20)
        return tmp;

    //Compute Affine Matrix
    affine = estimateRigidTransform(RS_Features1, RS_Features2, false);
    //affine = estimateRigidTransform(keypoints_1, keypoints_2, false);
    

    if(affine.empty() == 1)
        return tmp;

    aff[3] = affine.at<double>(0,2);
    aff[4] = affine.at<double>(1,2);
    aff[2] = atan2(affine.at<double>(1,0), affine.at<double>(0,0));
    aff[0] = affine.at<double>(0,0)/cos(da);
    aff[1] = affine.at<double>(1,1)/cos(da);

    
    return aff;
}

//Motion Compensate
//Input: PreviousFrame and CurrentFrame in grayscale
//Output: Stabilized CurrentFrame
Mat VideoStab::Compensate(Mat frame1 , Mat frame2, vector<double> &aff, vector<double> &sum)
{
    int vert_border = HORIZONTAL_BORDER_CROP * frame1.rows / frame1.cols;

    if(k==1)
        k++;
    else
    {
        Kalman_Filter(&thetha , &transX , &transY, sum);

        
        KF_thetha.push_back(thetha);
        KF_transX.push_back(transX);
        KF_transY.push_back(transY);
   
        
        GaussianBlur(KF_thetha, KF_thetha, Size(11,1), sigma, sigma);
        GaussianBlur(KF_transX, KF_transX, Size(11,1), sigma, sigma);
        GaussianBlur(KF_transY, KF_transY, Size(11,1), sigma, sigma);

       
        thetha = KF_thetha.back();
        transX = KF_transX.back();
        transY = KF_transY.back();

        Kalman_Filter(&thetha , &transX , &transY, sum);

    }

    
    diff_transX = transX - sum[3];
    diff_transY = transY - sum[4];
    diff_thetha = thetha - sum[2];

    aff[3] = aff[3] + diff_transX;
    aff[4] = aff[4] + diff_transY;
    aff[2] = aff[2] + diff_thetha;

    //Creating the smoothed parameters matrix
    smoothedMat.at<double>(0,0) = aff[0] * cos(aff[2]);
    smoothedMat.at<double>(0,1) = aff[0] * -sin(aff[2]);
    smoothedMat.at<double>(1,0) = aff[1] * sin(aff[2]);
    smoothedMat.at<double>(1,1) = aff[1] * cos(aff[2]);

    smoothedMat.at<double>(0,2) = aff[3];
    smoothedMat.at<double>(1,2) = aff[4];

    //Warp the new frame using the smoothed parameters
    warpAffine(frame1, smoothedFrame, smoothedMat, frame2.size());
    
    //Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    smoothedFrame = smoothedFrame(Range(vert_border, smoothedFrame.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols-HORIZONTAL_BORDER_CROP));

    resize(smoothedFrame, smoothedFrame, frame2.size());
        
    return smoothedFrame;
}


//Kalman Filter
void VideoStab::Kalman_Filter(double *thetha , double *transX , double *transY, vector <double> &sum)
{
    
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;
    
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;
    
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);
    
    *thetha = frame_1_thetha + gain_thetha * (sum[2] - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum[3] - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum[4] - frame_1_transY);

    errthetha = ( 1 - gain_thetha ) * frame_1_errthetha;
    errtransX = ( 1 - gain_transX ) * frame_1_errtransX;
    errtransY = ( 1 - gain_transY ) * frame_1_errtransY;
}

//---------------------DeMotionblurAlgorithm----------------------


void VideoStab::i2z(cv::Mat src, cv::Mat& dst)
{
	//convert the image to float type, create another one filled with zeros, 
	//and make an array of these 2 images
	cv::Mat im_array[] = { cv::Mat_<float>(src), cv::Mat::zeros(src.size(), CV_32F) };

	//combine as a 2 channel image to represent a complex number type image
	cv::Mat im_complex; cv::merge(im_array, 2, im_complex);
 
    //copy to destination
	im_complex.copyTo(dst);
}
 

void VideoStab::z2i(cv::Mat src, cv::Mat& dst)
{
    //split the complex image to 2
	cv::Mat im_tmp[2]; cv::split(src, im_tmp);
    
    //get absolute value
	cv::Mat im_f; cv::magnitude(im_tmp[0], im_tmp[1], im_f);
    
    //copy to destination
	im_f.copyTo(dst);
}
 
Mat VideoStab::complexDiv(const cv::Mat& A, const cv::Mat& B)
{
    cv::Mat A_tmp[2]; cv::split(A, A_tmp);
    cv::Mat a, b;
    A_tmp[0].copyTo(a);
    A_tmp[1].copyTo(b);
    
    cv::Mat B_tmp[2]; cv::split(B, B_tmp);
    cv::Mat c, d;
    B_tmp[0].copyTo(c);
    B_tmp[1].copyTo(d);
    
    cv::Mat C_tmp[2];
    cv::Mat g = (c.mul(c)+d.mul(d));
    C_tmp[0] = (a.mul(c)+b.mul(d))/g;
    C_tmp[1] = (b.mul(c)-a.mul(d))/g;
    
    cv::Mat C;
    cv::merge(C_tmp, 2, C);
 
    return C;
}

void VideoStab::genaratePsf(Mat &psf, double len,double angle)
{
	double half=len/2;											
	double alpha = (angle-floor(angle/ 180) *180) /180* 3.14;
	double cosalpha = cos(alpha);
	double sinalpha = sin(alpha);
	int xsign;
	if (cosalpha < 0)
	{
		xsign = -1;
	}
	else
	{
		if (angle == 90)
		{
			xsign = 0;
		}
		else
		{
			xsign = 1;
		}
	}
	int psfwdt = 1;
	int sx = (int)fabs(half*cosalpha + psfwdt*xsign - len*CV_TERMCRIT_EPS);
	int sy = (int)fabs(half*sinalpha + psfwdt - len*CV_TERMCRIT_EPS);
	Mat_<double> psf1(sy, sx, CV_64F);
	Mat_<double> psf2(sy * 2, sx * 2, CV_64F);
	int row = 2 * sy;
	int col = 2 * sx;
	
	for (int i = 0; i < sy; i++)
	{
		double* pvalue = psf1.ptr<double>(i);
		for (int j = 0; j < sx; j++)
		{
			pvalue[j] = i*fabs(cosalpha) - j*sinalpha;
 
			double rad = sqrt(i*i + j*j);
			if (rad >= half && fabs(pvalue[j]) <= psfwdt)
			{
				double temp = half - fabs((j + pvalue[j] * sinalpha) / cosalpha);
				pvalue[j] = sqrt(pvalue[j] * pvalue[j] + temp*temp);
			}
			pvalue[j] = psfwdt + CV_TERMCRIT_EPS - fabs(pvalue[j]);
			if (pvalue[j] < 0)
			{
				pvalue[j] = 0;
			}
		}
	}

	for (int i = 0; i < sy; i++)
	{
		double* pvalue1 = psf1.ptr<double>(i);
		double* pvalue2 = psf2.ptr<double>(i);
		for (int j = 0; j < sx; j++)
		{
			pvalue2[j] = pvalue1[j];
		}
	}
 
	for (int i = 0; i < sy; i++)
	{
		for (int j = 0; j < sx; j++)
		{
			psf2[2 * sy -1 - i][2 * sx -1 - j] = psf1[i][j];
			psf2[sy + i][j] = 0;
			psf2[i][sx + j] = 0;
		}
	}

	double sum = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum+= psf2[i][j];
		}
	}
	psf2 = psf2 / sum;
	if (cosalpha>0)
		flip(psf2, psf2, 0);
 
	psf = psf2;

}
 
/*Mat VideoStab::AddMotionBlur(const cv::Mat& src, const cv::Mat& ker)  
{   
    // convert to float data
    cv::Mat sample_float;
    src.convertTo(sample_float, CV_32FC1);
    
    // motion blur
    cv::Point anchor(0, 0);
    double delta = 0;
    cv::Mat dst = cv::Mat(sample_float.size(), sample_float.type());
    Ptr<cv::FilterEngine> fe = cv::createLinearFilter(sample_float.type(), ker.type(), ker, anchor,
        delta, BORDER_WRAP, BORDER_CONSTANT, cv::Scalar(0));
    fe->apply(sample_float, dst);
    
    return dst; 
}*/


Mat VideoStab::Deblur(Mat input)
{
    Mat input_, tmp_I;
    i2z(input,tmp_I);
    dft(tmp_I,tmp_I);
    Mat tmp_I2(tmp_I.size(),tmp_I.type());

    Mat output(input.size(),input.type()), output_, tmp_O;

    int minIdx[2]={255,255},maxIdx[2]={255,255};
    double minv,maxv;


    for(int i=0;i<input.rows;i++)
        for(int j=0;j<input.cols;j++)
            {
            int a=tmp_I.at<Vec3b>(i,j)[0];
            int b=tmp_I.at<Vec3b>(i,j)[1];
            tmp_I2.at<Vec3b>(i,j)[0]=sqrt(pow(a,2)+pow(b,2))+1;
            tmp_I2.at<Vec3b>(i,j)[1]=1;
            }

    log(tmp_I2,tmp_I2);
    for(int i=0;i<input.rows;i++)
        for(int j=0;j<input.cols;j++)
            {
            tmp_I2.at<Vec3b>(i,j)[1]=0;
            }

    idft(tmp_I2,tmp_I2);
    z2i(tmp_I2,input_);

    minMaxIdx(input_,&minv,&maxv,minIdx,maxIdx);
    minIdx[0]=input.rows - minIdx[0] - 1;
    minIdx[1]=input.cols - minIdx[1] - 1;

    double d=3;
    double fi=0.5;
    //double fi=atan(minIdx[0]/minIdx[1])*180/3.14+0.5;
    //double d=sqrt(pow(minIdx[0],2) + pow(minIdx[1],2))/2;


    Mat PSF, PSF_, tmp_P;

    /*for(int i=0;i<input.rows;i++)
        for(int j=0;j<input.cols;j++)
            if(sqrt(pow(i-input.rows/2+0.5,2) + pow(j-input.cols/2+0.5,2)) <= d)
                PSF.at<float>(i,j) = 1/d;
            else
                PSF.at<float>(i,j) = 0;*/
    float kernel[1][3] = {{0.333333333,0.33333333,0.33333333}};
    PSF=Mat(1, 3, CV_32FC1, &kernel);
    //genaratePsf(PSF,d,fi);


    //cv::imwrite("/home/de/webwxgetmsgimg3.jpeg", PSF);

    Mat tmp = Mat::zeros(input.rows, input.cols, CV_32FC1);
    PSF.copyTo(tmp(cv::Rect(0,0,PSF.cols,PSF.rows)));

    i2z(tmp,tmp_P);
    dft(tmp_P,PSF_);
    output_=complexDiv(tmp_I,PSF_);

    dft(output_, tmp_O,DFT_INVERSE+DFT_SCALE); 
    z2i(tmp_O, output);
    
    
    return input;
}


