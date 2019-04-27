

#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat frame_test;
    frame_test = cap.open("/home/chenxin/data/test.mp4");
    VideoCapture cap_add;
    Mat frame_add;
    frame_add = cap_add.open("/home/chenxin/source/AR-with-OpenCV/SurfAR/xiaohuangren.mp4");
    Mat img_object = imread("/home/chenxin/source/AR-with-OpenCV/SurfAR/SLAM.png");     //目标图像
    //inital
    Mat descriptor_object;
    std::vector<KeyPoint> keypoint_object;
    Mat descriptor_scene;
    std::vector<KeyPoint> keypoint_scene;
    Mat out_matches;    //定义匹配图像用于输出

    BFMatcher matcher (NORM_HAMMING);
    Mat H_latest = Mat::eye(3, 3, CV_32F); //指定大小和类型的单位矩阵H_latest，可对其进行缩放操作.
    Mat scene_mask;  //= Mat::zeros(img_scene.rows, img_scene.cols, CV_8UC1);  //定义一个与场景图像大小一样的掩膜scene_mask.      
    
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    //detect
    orb->detect(img_object, keypoint_object);
    //compute
    orb->compute(img_object, keypoint_object, descriptor_object);

    
    // if the mp4 is opened
    if(!cap.isOpened())
    {
        cout<<"can't open the picture"<<endl;
        return -1;
    }
    namedWindow("output",0);
    resizeWindow("output",640,480);
    
    while(cap.read(frame_test))
    {
        bool good_detection = false;
        scene_mask = Mat::zeros(frame_test.rows, frame_test.cols, CV_8UC1);       //初始化掩膜.!!!!!!!!!!别忘了
        if(!frame_test.empty())
        {
            //detect
            orb->detect(frame_test, keypoint_scene);
            //compute
            orb->compute(frame_test, keypoint_scene, descriptor_scene);
            std::vector<DMatch> matches;
            matcher.match(descriptor_object, descriptor_scene, matches); 
            
            // good_matches
            double max_dist = 0;
            double min_dist = 100;
            
            for(int i = 0;i < descriptor_object.rows;i++)
            {
                double dist = matches[i].distance;
                if(dist < min_dist)
                    min_dist = dist;
                if(dist > max_dist)
                    max_dist = dist;
            }
            
            std::vector<DMatch> good_matches;
            
            for(int i = 0;i < descriptor_object.rows;i++)
            {
                if(matches[i].distance <= 4*min_dist)
                    good_matches.push_back(matches[i]);
            }
            //if the matches is few.
            if (keypoint_scene.size() < 10)
                continue;
            drawMatches(img_object, keypoint_object, frame_test, keypoint_scene, good_matches, out_matches);    
            
            
            
            //++++++++++++++++++++++++++++++++++++++++++++++++++DLT+++++++++++++++++++++++++++++++++++++++++++++
            if(out_matches.data)//?????what the condition is???????
            {
                // ====================================【获取两幅图像特征点的坐标】======================================START
                std::vector<Point2f> obj;
                std::vector<Point2f> scene;
                for( size_t i = 0; i < good_matches.size(); i++ )
                {
                    //-- Get the keypoints from the good matches
                    //good_matches[i].queryIdx保存目标图像匹配点的序号
                    //good_matches[i].trainIdx保存相机获取图像的匹配点的序号
                    obj.push_back( keypoint_object[ good_matches[i].queryIdx ].pt );   //～.pt 为该序号对应的点的坐标，～.pt.x为该点的x坐标.
                    scene.push_back( keypoint_scene[ good_matches[i].trainIdx ].pt );  //
                }
//                //输出目标图像和特征点的坐标
//                for(auto o :obj)
//                    std::cout << o << " ";
//                std::cout << "++++++++" << std::endl;
                // ====================================【获取两幅图像特征点的坐标】=======================================END

                // ====================================【计算变换矩阵和角点坐标】=======================================START
                //Homography是一个变换（3*3矩阵），将一张图中的点映射到另一张图中对应的点.
                //四对以上的匹配点即能够计算出变换矩阵H.
                Mat H = findHomography( obj, scene, RANSAC );
                if (!H.data)
                {
                    continue;
                }

                // ===================【获取目标图像的四个角的坐标】（若不绘制轮廓可删除）============================START
                std::vector<Point2f> obj_corners(4);
                obj_corners[0] = cvPoint(0, 0);
                obj_corners[1] = cvPoint( img_object.cols, 0 );
                obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
                obj_corners[3] = cvPoint( 0, img_object.rows );
//                //输出四个角的坐标分别为[0, 0] [671, 0] [671, 960] [0, 960]
//                for(auto o :obj_corners)
//                    std::cout << o << " ";
//                std::cout << "++++++++" << std::endl;

                std::vector<Point2f> scene_corners(4);
                //scene_corners与输入obj_corners大小相同，H时3*3的浮点型变换矩阵.
                perspectiveTransform( obj_corners, scene_corners, H);
//                //输出场景的角点坐标[128.488, 4.11725] [415.675, -62.3487] [368.943, 441.185] [42.9645, 424.308]
//                for(auto s :scene_corners)
//                    std::cout << s << " ";
//                std::cout << "++++++++" << std::endl;
                // ===================【获取目标图像的四个角的坐标】（若不绘制轮廓可删除）=============================EEND

                // 检查转换矩阵，变换是否合理.
                float hDet = abs(determinant(H));       //determinant()返回矩阵H的行列式.
                if (hDet < 100 && hDet > 0.05)
                { // Good detection, reasonable transform
                    H_latest = H;
                    good_detection = true;
                }
                // ====================================【计算变换矩阵和角点坐标】=======================================END

                // ================================【绘制目标图像在场景中的轮廓，不绘制可删除】===============================START
                std::vector<Point2f> match_corners(4);
                match_corners[0] = scene_corners[0] + Point2f( img_object.cols, 0);
                match_corners[1] = scene_corners[1] + Point2f( img_object.cols, 0);
                match_corners[2] = scene_corners[2] + Point2f( img_object.cols, 0);
                match_corners[3] = scene_corners[3] + Point2f( img_object.cols, 0);

                line( out_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
                line( out_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( out_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                line( out_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
                //-- Show detected matches
               // imshow( "目标轮廓", out_matches );
                // ================================【绘制目标图像在场景中的轮廓，不绘制可删除】===============================END

//                 Mat img_video;
//                 vid >> img_video;
                cap_add.read(frame_add);
                if(frame_add.empty())
                {
                    std::cout << "视频结束" << std::endl;
                    break;
                }
                resize(frame_add, frame_add, Size(frame_test.cols,frame_test.rows));

                // 在场景中为需要叠加的视频创建掩膜.
                std::vector<Point2f> vid_corners(4);        //叠加的视频坐标信息vid_corners
                vid_corners[0] = cvPoint( 0, 0 );
                vid_corners[1] = cvPoint( frame_add.cols, 0 );
                vid_corners[2] = cvPoint( frame_add.cols, frame_add.rows );
                vid_corners[3] = cvPoint( 0, frame_add.rows );

                //对掩膜进行仿射变换.
                cv::Point nonfloat_corners[4];
                for(int i=0; i<4; i++)
                {
                    nonfloat_corners[i] = vid_corners[i];
                }
                fillConvexPoly(scene_mask, nonfloat_corners, 4, cv::Scalar(255));   //绘制一个掩膜scene_mask，与场景角点坐标一致
                warpPerspective( scene_mask, scene_mask, H_latest, Size(frame_test.cols,frame_test.rows));    //warpPerspective透视变换

                //加入下面这段代码后运行一段时间崩溃.
                //对叠加对象（视频）进行仿射变换.
                warpPerspective( frame_add, frame_add, H_latest, Size(frame_test.cols,frame_test.rows));

                //将视频帧复制到场景中.
                if(good_detection)
                {
                    frame_add.copyTo(frame_test, scene_mask);
                }

                imshow( "output", frame_test);

                waitKey(1);
            }    
            waitKey(30);
        }
    }
    return 0;
}
