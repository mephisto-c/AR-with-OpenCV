/*
 
 
 */
#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<cmath>


using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
    
    //read an image
    Mat img_object = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/state.jpg",CV_LOAD_IMAGE_COLOR);
    Mat img_object_1 = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/state2.jpg",CV_LOAD_IMAGE_COLOR);
    
    
    //inital
    Mat descriptor;
    std::vector<KeyPoint> keypoint;
    Mat descriptor_1;
    std::vector<KeyPoint> keypoint_1;
    
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    
    //detect
    orb->detect(img_object, keypoint);
    orb->detect(img_object_1, keypoint_1);
    
    //compute
    orb->compute(img_object, keypoint, descriptor);
    orb->compute(img_object_1, keypoint_1, descriptor_1);
    
    
    //draw
    Mat outing, outing_1;
    drawKeypoints(img_object, keypoint, outing, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_object_1, keypoint_1, outing_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    //good_matches
    //FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    BFMatcher matcher (NORM_HAMMING);
    matcher.match(descriptor, descriptor_1, matches);
    
    double max_dist = 0;
    double min_dist = 100;
    
    for(int i = 0;i < descriptor.rows;i++){
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }
    
    std::vector<DMatch> good_matches;
    for(int i = 0;i < descriptor.rows;i++){
        if(matches[i].distance <= 4*min_dist)
            good_matches.push_back(matches[i]);
    }
    
    //show
    Mat img_matches, img_good_matches;
    drawMatches(img_object, keypoint, img_object_1, keypoint_1, matches, img_matches);
    drawMatches(img_object, keypoint, img_object_1, keypoint_1, good_matches, img_good_matches);
    imshow("matches",img_matches);
    imshow("good_matches",img_good_matches);
    waitKey(0);
    destroyAllWindows();
    
    return 0;
    
    
}
