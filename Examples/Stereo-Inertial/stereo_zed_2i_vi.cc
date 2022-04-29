/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

#include<json.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<uint64_t> &vTimestamps);

void LoadIMU(const string &strPathToSequence, vector<cv::Point3f> &acclData,
             vector<cv::Point3f> &gyroData, vector<uint64_t> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4) {
        cerr << endl << "Usage: ./stereo_zed_2i_vi path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<cv::Point3f> vAcc;
    vector<cv::Point3f> vGyr;
    vector<uint64_t> vTimestampsImu;
    vector<uint64_t> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);
    LoadIMU(string(argv[3]), vAcc, vGyr, vTimestampsImu);
    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    double t_track = 0.f;
    double t_resize = 0.f;

    // Main loop
    cv::Mat imLeft, imRight;

    size_t last_imu_idx = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni]*1e-9;

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = imLeft.cols * imageScale;
            int height = imLeft.rows * imageScale;
            cv::resize(imLeft, imLeft, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC14
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Load imu measurements from previous frame
        vImuMeas.clear();
        while(vTimestampsImu[last_imu_idx]*1e-9 <= tframe && tframe > 0)
        {
            vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[last_imu_idx].x,vAcc[last_imu_idx].y,vAcc[last_imu_idx].z,
                                                     vGyr[last_imu_idx].x,vGyr[last_imu_idx].y,vGyr[last_imu_idx].z,
                                                     vTimestampsImu[last_imu_idx]*1e-9));
            last_imu_idx++;
        }
        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]*1e-9-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1]*1e-9;

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

void LoadIMU(const string &strPathToSequence, vector<cv::Point3f> &acclData,
             vector<cv::Point3f> &gyroData, vector<uint64_t> &vTimestamps) {

    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/sensor_data.json";
    fTimes.open(strPathTimeFile.c_str());
    std::ifstream in(strPathTimeFile);
    nlohmann::json sensor_data;
    in >> sensor_data;

    for(const auto& it : sensor_data["angular_velocity"].items()) {
        const cv::Point3f gyro = cv::Point3f(
                    M_PI/180.f*(float)sensor_data["angular_velocity"][it.key()][0],
                    M_PI/180.f*(float)sensor_data["angular_velocity"][it.key()][1],
                    M_PI/180.f*(float)sensor_data["angular_velocity"][it.key()][2]);
        gyroData.push_back(gyro);
        const uint64_t t_ns = std::stod(it.key());
        vTimestamps.push_back(t_ns*1e-9);
    }

    for(const auto& it : sensor_data["linear_acceleration"].items()) {
        const cv::Point3f accl = cv::Point3f(
                    sensor_data["linear_acceleration"][it.key()][0],
                    sensor_data["linear_acceleration"][it.key()][1],
                    sensor_data["linear_acceleration"][it.key()][2]);
        acclData.push_back(accl);
    }
}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<uint64_t> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/test_poses.json";
    fTimes.open(strPathTimeFile.c_str());
    std::ifstream in(strPathTimeFile);
    nlohmann::json cam_poses;
    in >> cam_poses;

    for(const auto& it : cam_poses["camera_pose"].items()) {
        const uint64_t t_ns = std::stoll(it.key());
        vTimestamps.push_back(t_ns);
        std::cout << it.key() << " : " << t_ns << "\n";

    }

    string strPrefixLeft = strPathToSequence + "/test_images/left/left_";
    string strPrefixRight = strPathToSequence + "/test_images/right/right_";


    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + std::to_string(vTimestamps[i]) + ".png";
        vstrImageRight[i] = strPrefixRight + std::to_string(vTimestamps[i]) + ".png";
    }
}
