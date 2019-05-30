#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <time.h>
#include <stablization.h>
#include <omp.h>
#include <condition_variable>
#include <unistd.h>

using namespace std;
using namespace cv;

long double Metric(Mat frame1, Mat frame2);
bool Judge(vector<double> &tmp);

const int HORIZONTAL_BORDER_CROP = 30;

int main(int argc, char **argv)
{
    string video_name;
    cin >> video_name;    

    VideoCapture cap;
    //cout << cap.get(CV_CAP_PROP_FPS) << endl;
    cap.open("/home/de/" + video_name); //if you wanna open a local video file 
    if (!cap.isOpened()){
        cerr << "ERROR: Can't open such video!" << endl;
		exit(1);}

    VideoStab stab;

    register Mat frame_2, frame2, frame_c2;
    register Mat frame_1, frame1, frame_c1;
    queue <Mat> frame;
    register Mat Stabframe1, Stabframe2;


    register queue <vector<double>> sum_;
    vector<double> sum = {0, 0, 0, 0, 0};
    register queue <vector<double>> aff_;
    vector<double> aff;


    double t, s;
    double fps, fps2, fps_average = 0;

    double ITF1 = 0,ITF2 = 0;
    register double count = 0;
    register double count_ = 1;
    
    mutex m;
    //unique_lock<std::mutex> lk(m);

    cap >> frame_1;
    cvtColor(frame_1, frame1, COLOR_BGR2GRAY);
    frame_c1 = frame1;


    VideoWriter outputVideo;
    outputVideo.open(video_name + ".avi" , CV_FOURCC('X' , 'V' , 'I' , 'D'), 100 , Size(frame1.rows, frame1.cols*2+10));

    #pragma omp parallel sections num_threads(2)  
    {
    
    while(1)
    {
       s = (double)cv::getTickCount();
        ++count;

        cap >> frame_2;
        if(frame_2.data == NULL)
            break;

        cvtColor(frame_2, frame2, COLOR_BGR2GRAY);
        
        
        //ITF1 = ITF1*((count-1)/count) + Metric(frame1, frame2)/count;  //ITF for origin video sequence

        aff = stab.Estimate(frame1 , frame2);
        aff_.push(aff);

        for (int i = 0; i < 5; i++)
            sum[i] += aff[i];
        sum_.push(sum);
        frame.push(frame2);

        frame1 = frame2.clone();

        s = ((double)cv::getTickCount() - s) / cv::getTickFrequency();
        fps2 = 1.0 / s;
        //cout << fps2 << endl;


    }

    #pragma omp section 
    {
    usleep(110000);
    while(1)
    {
        t = (double)cv::getTickCount();
        if (frame.empty())
        {
        usleep(30000);
            
            if (frame.empty())
                    break;
            else 
                continue;
        }
        

        frame_c2 = frame.front();
        frame.pop();

        Stabframe2 = stab.Compensate(frame_c1 , frame_c2, aff_.front(), sum_.front());

        aff_.pop();
        sum_.pop();

        //ITF2 = ITF2*((count-1)/count) + Metric(Stabframe1, Stabframe2)/count;  //ITF for stabilized video sequence

        Mat Result = Mat::zeros(frame_c2.rows, frame_c2.cols*2+10, frame_c2.type());

        frame_c2.copyTo(Result(Range::all(), Range(0, Stabframe2.cols)));

        Stabframe2.copyTo(Result(Range::all(), Range(Stabframe2.cols+10, Stabframe2.cols*2+10)));

        if(Result.cols > 1920)
            resize(Result, Result, Size(Result.cols/2, Result.rows/2));
        
       
        frame_c1 = frame_c2.clone();
        
        waitKey(3);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
       
        putText(Result, "FPS: " + to_string(fps), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); 
        imshow("Result", Result);
        fps_average = fps_average * ((count_-1) / count_) + fps / count_;
        cout << count_ << "   "<< frame.size() <<endl;
        ++count_;
    }
    cout << "Abortion!" <<endl;
    }
    }
    cout <<"Average FPS = " << fps_average << endl;
    //cout<<"ITF1 = "<<ITF1<<endl;  //original video average peak signal-noise-ratio
    //cout<<"ITF2 = "<<ITF2<<endl;  //stabilized video average peak signal-noise-ratio

    return 0;
}



//Compute PSNR beteen 2 adjacent frames
long double Metric(Mat frame1, Mat frame2)
{
    int height = frame1.rows;
    int width = frame1.cols;
    long double MSE = 0, PSNR = 0;

    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            {
                long double value1 = frame1.at<uchar>(i,j);
                long double value2 = frame2.at<uchar>(i,j);
                MSE += (value1 - value2) / (height * width) * (value1 - value2);
            }

    PSNR = 10 * log10(255 * 255 / MSE);

    return PSNR;
}

bool Judge(vector<double> &tmp)
{
    cout << (tmp[0]>1.0) << (tmp[1]>1.0) << (tmp[2]>0.001) <<endl;
    
    if(tmp[0]>1.0 || tmp[1]>1.0 || tmp[2]>0.001)
        return 1;
    else return 0;

}