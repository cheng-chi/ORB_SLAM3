
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

#include "GCNExtractor.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    ORB_SLAM3::GCNExtractor gcn_extractor(
                1000, 1.2, 8, 10, 10, "/home/steffen/Dokumente/ORB_SLAM3/Models/kp2d_resnet_traced_320_256.pt");

    cv::Mat I1 = cv::imread("/home/steffen/Dokumente/data/TestMappingRelocalization/img001.jpg");
    cv::Mat I2 = cv::imread("/home/steffen/Dokumente/data/TestMappingRelocalization/img002.jpg");

    //cv::cvtColor(image_now, image_now, cv::COLOR_BGR2RGB);
    cv::Mat descriptors1,descriptors2;
    vector<cv::KeyPoint> keypoints1,keypoints2;
    gcn_extractor.operator()(I1, cv::Mat(), keypoints1, descriptors1);
    gcn_extractor.operator()(I2, cv::Mat(), keypoints2, descriptors2);

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


    return 0;
}

