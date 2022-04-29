/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#include "KP2DExtractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>



using namespace cv;
using namespace std;

namespace ORB_SLAM3
{
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


void nms(const cv::Mat& det, const cv::Mat& desc, const cv::Mat& score,
         std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){

    std::vector<cv::Point2f> pts_raw;
    std::vector<float> score_raw;
    for (int i = 0; i < det.rows; i++){

        float u = det.at<float>(i, 0);
        float v = det.at<float>(i, 1);
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

    descriptors.create(select_indice.size(), desc.cols, CV_8UC1);

    for (size_t i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j<desc.cols; j++)
        {
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}

KP2DExtractor::KP2DExtractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST, std::string path_to_model):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      //module = torch::jit::load(path_to_model);
      path_to_model_ = path_to_model;
      //module.to(torch::kCUDA);
      //module.eval();
      std::cout<<"Loaded deep feature extractor from "<<path_to_model<<"\n";
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
    }

}

torch::Tensor matToTensor(const cv::Mat &image)
{
    bool isChar = (image.type() & 0xF) < 2;
    std::vector<int64_t> dims = {image.rows, image.cols, image.channels()};
    return torch::from_blob(image.data, dims, isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
}

void KP2DExtractor::detect(const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints,
                           cv::Mat& descriptors, float conf_thresh)
{ 
    std::cout<<"Loading Model from "<<path_to_model_<<"\n";
    torch::jit::script::Module module = torch::jit::load(path_to_model_);
    module.to(torch::kCUDA);
    module.eval();
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);
    if(image.empty())
        return;

    assert(image.type() == CV_8UC3);

    const int img_org_width = image.cols;
    const int img_org_height = image.rows;
    const int img_width = 320;
    const int img_height = 240;
    const float ratio_width = float(img_org_width) / float(img_width);
    const float ratio_height = float(img_org_height) / float(img_height);
    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(img_width, img_height));

    cv::imshow("Image in detector", resized_img);
    cv::waitKey(1);
    cv::Mat img;
    resized_img.convertTo(img, CV_32FC3, 1.f / 255.f , 0);

    const int border = 5;
    const int dist_thresh = 5;

    auto img_var = matToTensor(img);

    img_var = img_var.permute({2,0,1});
    img_var.unsqueeze_(0);
    img_var.sub_(0.5);
    img_var.mul_(0.225);
    // to gray
    img_var = img_var.mean(1);
    img_var.unsqueeze_(1);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(device));
    auto output = module.forward(inputs).toTuple();
    auto desc = output->elements()[2].toTensor();
    auto desc_size = desc.sizes();
    const auto C = desc_size[1];
    const auto Hc = desc_size[2];
    const auto Wc = desc_size[3];

    auto score_  = output->elements()[0].toTensor().view({1, Hc, Wc}).view({1, -1}).t().to(torch::kCPU).squeeze();
    auto coord_ = output->elements()[1].toTensor().view({2, Hc, Wc}).view({2, -1}).t().to(torch::kCPU).squeeze();
    auto desc_ = desc.view({C, Hc, Wc}).view({C, -1}).t().to(torch::kCPU).squeeze();

    auto boolean_selector = (score_ > conf_thresh).squeeze();
    auto score_f = score_.index({boolean_selector});
    auto coord_f = coord_.index({boolean_selector, torch::indexing::Slice()});
    auto desc_f = desc_.index({boolean_selector, torch::indexing::Slice()});

    const auto nr_pts = score_f.sizes()[0];

    cv::Mat pts_mat = cv::Mat(nr_pts, 2, CV_32FC1, coord_f.data_ptr<float>());
    cv::Mat score_mat = cv::Mat(nr_pts, 1, CV_32FC1, score_f.data_ptr<float>());
    cv::Mat desc_mat = cv::Mat(nr_pts, C, CV_32FC1, desc_f.data_ptr<float>());

    cv::Mat final_desc;
    final_desc.create(nr_pts, int(C/8), CV_8UC1);

    int bit;
    for (int k = 0; k < nr_pts; ++k) {
        unsigned int val = 0;
        int idx = 0;
        float* dsc_ptr = desc_mat.ptr<float>(k);
        for (int i = 0; i < C; i++){
            bit = i%8;
            int b = dsc_ptr[i];
            if (bit == 0) {
                val = b;
            } else {
                val |= b << bit;
            }
            if (bit == 7) {
                final_desc.at<unsigned char>(k, idx) = val;
                val = 0;
                idx++;
            }
        }
    }
    nms(pts_mat, final_desc, score_mat, keypoints, descriptors,
        border, dist_thresh, img_width, img_height, ratio_width, ratio_height);
}

} //namespace ORB_SLAM

