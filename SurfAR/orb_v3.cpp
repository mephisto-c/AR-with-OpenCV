/*
 
 
 */
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;

// int main(int argc, char** argv)
// {
//     
//     //read an image
//     Mat img_object  = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/state.jpg",CV_LOAD_IMAGE_COLOR);
//     
//     //inital
//     Mat descriptor;
//     std::vector<KeyPoint> keypoint;
//     Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
//     
//     //detect
//     orb->detect(img_object, keypoint);
//     
//     //compute
//     orb->compute(img_object, keypoint, descriptor);
//     
//     //draw
//     Mat outing;
//     drawKeypoints(img_object, keypoint, outing, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//     
//     imshow("ORB_feature", outing);
//     waitKey(0);
//     destroyAllWindows();
//     
//     return 0;
//     
//     
// }

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat frame_test;
    frame_test = cap.open("/home/chenxin/data/test.mp4");
    Mat img_object = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/SLAM.png");     //目标图像
    //inital
    Mat descriptor_object;
    std::vector<KeyPoint> keypoint_object;
    Mat descriptor_scene;
    std::vector<KeyPoint> keypoint_scene;
    Mat out_matches;     //定义匹配图像用于输出.

    
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    //detect
    orb->detect(img_object, keypoint_object);
    //compute
    orb->compute(img_object, keypoint_object, descriptor_object);
    if(!cap.isOpened())
    {
        cout<<"can't open the video"<<endl;
        return -1;
    }
    namedWindow("output",0);
    resizeWindow("output", 640, 480);
    while(cap.read(frame_test))
    {
        //detect
        orb->detect(frame_test, keypoint_scene);
        //compute
        orb->compute(frame_test, keypoint_scene, descriptor_scene);
        
        //come from v2_orb
        std::vector<DMatch> matches;
        BFMatcher matcher (NORM_HAMMING);
        matcher.match(descriptor_object, descriptor_scene, matches);
        
        double max_dist = 0;
        double min_dist = 100;
        
        for(int i = 0;i < descriptor_object.rows;i++){
            double dist = matches[i].distance;
            if(dist < min_dist)
                min_dist = dist;
            if(dist > max_dist)
                max_dist = dist;
        }
        
        std::vector<DMatch> good_matches;
        for(int i = 0;i < descriptor_object.rows;i++){
            if(matches[i].distance <= 4*min_dist)
                good_matches.push_back(matches[i]);
        }
        
        // 【7】显示结果.
        //当特征点太少时不调用drawMatches（否则会崩溃）.
        if (keypoint_scene.size() < 10)
            continue;
        drawMatches(img_object, keypoint_object, frame_test, keypoint_scene, good_matches, out_matches);       
        imshow("output", out_matches);
        waitKey(1);
    }
    cap.release();
        
    return 0;
}
