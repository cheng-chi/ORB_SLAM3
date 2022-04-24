
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
using namespace torch::indexing;

void nms(const cv::Mat& det, const cv::Mat& desc, const cv::Mat& score,
         std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){

    std::vector<cv::Point2f> pts_raw;
    std::vector<float> score_raw;
    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
        score_raw.push_back(score.at<float>(i));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (size_t i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }

    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (size_t i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                grid.at<char>(vv + k, uu + j) = 0;

            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(
                                  pts_raw[select_ind].x * ratio_width,
                                  pts_raw[select_ind].y * ratio_height,
                                  1.0f, -1.f, score_raw[select_ind]));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }

    descriptors.create(select_indice.size(), 32, CV_8U);

    for (size_t i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j<32; j++)
        {
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}

void loadFeatures(vector<vector<cv::Mat > > &features, cv::Mat descriptors);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void createVocabularyFile(OrbVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat > > &features);

// ----------------------------------------------------------------------------
// Convert a char/float mat to torch Tensor
torch::Tensor matToTensor(const cv::Mat &image)
{
    bool isChar = (image.type() & 0xF) < 2;
    std::vector<int64_t> dims = {image.rows, image.cols, image.channels()};
    return torch::from_blob(image.data, dims, isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
}


int main(int argc, char *argv[])
{

    if(argc < 3)
    {
        cerr << endl << "Usage: ./create_vocabulary path_to_final_vocabulary path_to_imgs path_to_deep_model " << endl;
        return 1;
    }
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);

    ORB_SLAM3::GCNExtractor gcn_extractor(
                1000, 1.2, 8, 10, 10, argv[3]);

    vector<cv::String> fn;
    cv::glob(std::string(argv[2])+"/*.jpg", fn, false);

    //fn.push_back("/home/steffen/Downloads/val2017/000000000785.jpg");
    size_t count = fn.size(); 

    vector<vector<cv::Mat > > features;

    cout << "... Feature extraction started on "<<count<<" images!" << endl;

    int index = 0;
    for (size_t i=0; i<count; i++)
    {
        index++;

        if (index % 100) {
            std::cout<<"Extracted features from "<<index<<"/"<<count<<" images.\n";
        }
        if (index>0)
        {
            
            std::cout << fn[i] << std::endl;
            cv::Mat image_now = cv::imread(fn[i]);

            if (image_now.channels() < 3) {
                cv::cvtColor(image_now, image_now, cv::COLOR_GRAY2BGR);
            }
            if (image_now.cols < image_now.rows) {
                cv::rotate(image_now, image_now, cv::ROTATE_90_CLOCKWISE);
            }

            int img_width = 320;
            int img_height = 256;
            // avoid upsampling
            if (image_now.cols < img_width || image_now.rows < img_height) {
                continue;
            }
            cv::cvtColor(image_now, image_now, cv::COLOR_BGR2RGB);
            cv::Mat descriptors;
            vector<cv::KeyPoint> keypoints;
            gcn_extractor.operator()(image_now, cv::Mat(), keypoints, descriptors);
            loadFeatures(features, descriptors);
        }
    }

    cout << "... Extraction done!" << endl;

    // define vocabulary
    const int nLevels = 6;
    const int k = 10; // branching factor
    const DBoW2::WeightingType weight = DBoW2::TF_IDF;
    const DBoW2::ScoringType score = DBoW2::L1_NORM;
    OrbVocabulary voc(k, nLevels, weight, score);

    std::string vocName = std::string(argv[1]);
    createVocabularyFile(voc, vocName, features);

    cout << "--- THE END ---" << endl;

    return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, cv::Mat descriptors)
{
    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void createVocabularyFile(OrbVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat > > &features)
{

  cout << "> Creating vocabulary. May take some time ..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "> Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "> Saving vocabulary..." << endl;
  voc.saveToBinaryFile(fileName);
  cout << "... saved to file: " << fileName << endl;
}
// ----------------------------------------------------------------------------

