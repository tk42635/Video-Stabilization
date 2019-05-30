#ifndef VIDEOSTAB_H
#define VIDEOSTAB_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>


using namespace cv;
using namespace std;

class VideoStab
{
public:
    VideoStab();
    VideoCapture capture;

    Mat frame2;
    Mat frame1;

    int k;

    const int HORIZONTAL_BORDER_CROP = 30;

    Mat smoothedMat;
    Mat affine;

    Mat smoothedFrame, smoothedFrame_;

    double dx ;
    double dy ;
    double da ;
    double ds_x ;
    double ds_y ;

    double new_dx ;
    double new_dy ;
    double new_da ;
    double new_ds_x ;
    double new_ds_y ;

    double sx ;
    double sy ;

    double scaleX ;
    double scaleY ;
    double thetha ;
    double transX ;
    double transY ;

    double diff_scaleX ;
    double diff_scaleY ;
    double diff_transX ;
    double diff_transY ;
    double diff_thetha ;

    double errscaleX ;
    double errscaleY ;
    double errthetha ;
    double errtransX ;
    double errtransY ;

    double Q_scaleX ;
    double Q_scaleY ;
    double Q_thetha ;
    double Q_transX ;
    double Q_transY ;

    double R_scaleX ;
    double R_scaleY ;
    double R_thetha ;
    double R_transX ;
    double R_transY ;

    double sum_scaleX ;
    double sum_scaleY ;
    double sum_thetha ;
    double sum_transX ;
    double sum_transY ;

    
    vector <double> aff = {0, 0, 0, 0, 0};
    vector <double> tmp = {0, 0, 0, 0, 0};
    vector <vector<double>> tmp_;

    vector <double> KF_scaleX, KF_scaleY, KF_thetha, KF_transX, KF_transY;

    double ITF1,ITF2;

    vector<double> Estimate(Mat frame1 , Mat frame2);
    Mat Compensate(Mat frame1 , Mat frame2, vector<double> &aff, vector<double> &sum);
    void Kalman_Filter(double *thetha , double *transX , double *transY,  vector <double> &sum);

    void i2z(cv::Mat src, cv::Mat& dst);
    void z2i(cv::Mat src, cv::Mat& dst);
    cv::Mat complexDiv(const cv::Mat& A, const cv::Mat& B);
    void genaratePsf(Mat &psf, double len,double angle);
    Mat AddMotionBlur(const cv::Mat& src, const cv::Mat& ker);
    Mat Deblur(Mat input);
    
};

#endif // VIDEOSTAB_H
