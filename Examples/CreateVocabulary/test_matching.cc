
#include <iostream>
#include <vector>

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// DBoW2
#include "Thirdparty/DBoW2/DBoW2/DBoW2.h"

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "KP2DExtractor.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    ORB_SLAM3::KP2DExtractor kp2d_extractor(
                1000, 1.2, 8, 10, 10, "/media/Data/projects/ORB_SLAM3/KeypointModels/keypointnet_gray_model_traced_320_240.ckpt");

    cv::Mat I1 = cv::imread("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/test02.jpg");
    cv::Mat I2 = cv::imread("/media/Data/Sparsenet/OrbSlam3/TestMappingRelocalization/test03.jpg");

    //cv::cvtColor(image_now, image_now, cv::COLOR_BGR2RGB);
    cv::Mat descriptors1,descriptors2;
    vector<cv::KeyPoint> keypoints1,keypoints2;
    kp2d_extractor.detect(I1, cv::Mat(), keypoints1, descriptors1, 0.2);
    kp2d_extractor.detect(I2, cv::Mat(), keypoints2, descriptors2, 0.2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector< DMatch > matches;
    matcher->match( descriptors1, descriptors2, matches);
    //-- Draw matches
    Mat img_matches;
    drawMatches( I1, keypoints1, I2, keypoints2, matches, img_matches );
    //-- Show detected matches
    cv::resize(img_matches, img_matches, cv::Size(0,0), 0.5, 0.5);
    imshow("Good Matches", img_matches );
    waitKey();


    // test video
    cv::VideoCapture cap(argv[3]);
    // Check if camera opened successfully
    if (!cap.isOpened()) {
      std::cout << "Error opening video stream or file" << endl;
      return -1;
    }

    return 0;
}

