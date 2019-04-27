/*
 
 
 */
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    
    //read an image
    Mat img_object  = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/state.jpg",CV_LOAD_IMAGE_COLOR);
    
    //inital
    Mat descriptor;
    std::vector<KeyPoint> keypoint;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    
    //detect
    orb->detect(img_object, keypoint);
    
    //compute
    orb->compute(img_object, keypoint, descriptor);
    
    //draw
    Mat outing;
    drawKeypoints(img_object, keypoint, outing, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    
    imshow("ORB_feature", outing);
    waitKey(0);
    destroyAllWindows();
    
    return 0;
    
    
}
