/**
* This file is part of ORB-SLAM2.
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

#ifndef KP2DEXTRACTOR_H
#define KP2DEXTRACTOR_H

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// Compile will fail for opimizier since pytorch defined this
#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>


namespace ORB_SLAM3
{

class KP2DExtractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    KP2DExtractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST, std::string path_to_model);

    ~KP2DExtractor(){}


    void detect(const cv::Mat& image, const cv::Mat &mask, std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors, float conf_thresh=0.2f);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
    std::string path_to_model_;

};

} //namespace ORB_SLAM

#endif


