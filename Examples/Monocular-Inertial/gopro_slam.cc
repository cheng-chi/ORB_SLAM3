/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <signal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>

#include <json.h>
#include <CLI11.hpp>

using namespace std;
using nlohmann::json;
const double MS_TO_S = 1e-3; ///< Milliseconds to second conversion

void signal_callback_handler(int signum) {
   cout << "gopro_slam.cc Caught signal " << signum << endl;
   // Terminate program
   exit(signum);
}

bool LoadTelemetry(const string &path_to_telemetry_file,
                   vector<double> &vTimeStamps,
                   vector<double> &coriTimeStamps,
                   vector<cv::Point3f> &vAcc,
                   vector<cv::Point3f> &vGyro) {

    std::ifstream file;
    file.open(path_to_telemetry_file.c_str());
    if (!file.is_open()) {
      return false;
    }
    json j;
    file >> j;
    const auto accl = j["1"]["streams"]["ACCL"]["samples"];
    const auto gyro = j["1"]["streams"]["GYRO"]["samples"];
    const auto cori = j["1"]["streams"]["CORI"]["samples"];
    std::map<double, cv::Point3f> sorted_acc;
    std::map<double, cv::Point3f> sorted_gyr;

    for (const auto &e : accl) {
      cv::Point3f v((float)e["value"][0], (float)e["value"][1], (float)e["value"][2]);
      sorted_acc.insert(std::make_pair((double)e["cts"] * MS_TO_S, v));
    }
    for (const auto &e : gyro) {
      cv::Point3f v((float)e["value"][0], (float)e["value"][1], (float)e["value"][2]);
      sorted_gyr.insert(std::make_pair((double)e["cts"] * MS_TO_S, v));
    }

    double imu_start_t = sorted_acc.begin()->first;
    for (auto acc : sorted_acc) {
        vTimeStamps.push_back(acc.first-imu_start_t);
        vAcc.push_back(acc.second);
    }
    for (auto gyr : sorted_gyr) {
        vGyro.push_back(gyr.second);
    }
    for (const auto &e : cori) {
        coriTimeStamps.push_back((double)e["cts"] * MS_TO_S);
    }

    file.close();
    return true;
}

int main(int argc, char **argv) {
  // Register signal and signal handler
  // A process running as PID 1 inside a container 
  // is treated specially by Linux: it ignores any 
  // signal with the default action. As a result, 
  // the process will not terminate on SIGINT or 
  // SIGTERM unless it is coded to do so.
  // This allows stopping the docker container with ctrl-c
  signal(SIGINT, signal_callback_handler);

  // CLI parsing
  CLI::App app{"GoPro SLAM"};

  std::string vocabulary = "../../Vocabulary/ORBvoc.txt";
  app.add_option("-v,--vocabulary", vocabulary)->capture_default_str();

  std::string setting = "gopro10_maxlens_fisheye_setting_v1.yaml";
  app.add_option("-s,--setting", setting)->capture_default_str();

  std::string input_video;
  app.add_option("-i,--input_video", input_video)->required();

  std::string input_imu_json;
  app.add_option("-j,--input_imu_json", input_imu_json)->required();

  std::string output_trajectory_tum;
  app.add_option("--output_trajectory_tum", output_trajectory_tum);

  std::string output_trajectory_csv;
  app.add_option("-o,--output_trajectory_csv", output_trajectory_csv);

  std::string load_map;
  app.add_option("-l,--load_map", load_map);

  std::string save_map;
  app.add_option("--save_map", save_map);

  bool enable_gui = false;
  app.add_flag("-g,--enable_gui", enable_gui);

  int num_threads = 4;
  app.add_flag("-n,--num_threads", num_threads);

  std::string mask_img_path;
  app.add_option("--mask_img", mask_img_path);

  // Aruco tag for initialization
  int aruco_dict_id = cv::aruco::DICT_4X4_50;
  app.add_option("--aruco_dict_id", aruco_dict_id);

  int init_tag_id = 13;
  app.add_option("--init_tag_id", init_tag_id);

  float init_tag_size = 0.16; // in meters
  app.add_option("--init_tag_size", init_tag_size);

  // if lost more than max_lost_frames, terminate
  // disable the check if <= 0
  int max_lost_frames = -1;
  app.add_option("--max_lost_frames", max_lost_frames);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
      return app.exit(e);
  }

  cv::setNumThreads(num_threads);

  vector<double> imuTimestamps;
  vector<double> camTimestamps;
  vector<cv::Point3f> vAcc, vGyr;
  LoadTelemetry(input_imu_json, imuTimestamps, camTimestamps, vAcc, vGyr);

  // open setting to get image resolution
  cv::FileStorage fsSettings(setting, cv::FileStorage::READ);
  if(!fsSettings.isOpened()) {
     cerr << "Failed to open setting file at: " << setting << endl;
     exit(-1);
  }
  cv::Size img_size(fsSettings["Camera.width"],fsSettings["Camera.height"]);
  fsSettings.release();

  vector<double> vTimestamps;

  // load mask image
  cv::Mat mask_img;
  if (!mask_img_path.empty()) {
    mask_img = cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
    if (mask_img.size() != img_size) {
      std::cout << "Mask img size mismatch! Converting " << mask_img.size() << " to " << img_size << endl;
      cv::resize(mask_img, mask_img, img_size);
    }
  }

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(aruco_dict_id);
  ORB_SLAM3::System SLAM(
    vocabulary, setting, 
    ORB_SLAM3::System::IMU_MONOCULAR, 
    enable_gui, load_map, save_map,
    aruco_dict, init_tag_id, init_tag_size
  );

  // Open video file
  cv::VideoCapture cap(input_video, cv::CAP_FFMPEG);
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // Main loop
  int nImages = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);
  cout << "Video opened using backend " << cap.getBackendName() << endl;
  cout << "There are " << nImages << " frames in total" << endl;
  cout << "video FPS " << fps << endl;
  
  std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
  size_t last_imu_idx = 0;
  int n_lost_frames = 0;
  for (int frame_idx=0; frame_idx < nImages; frame_idx++){
    double tframe = (double)frame_idx / fps;

    // read frame from video
    cv::Mat im,im_track;
    bool success = cap.read(im);
    if (!success) {
      cout << "cap.read failed!" << endl;
      break;
    }

    // resize image and draw gripper mask
    im_track = im.clone();
    if (im_track.size() != img_size){
      cv::resize(im_track, im_track, img_size);
    }

    // apply mask image if loaded
    // if (!mask_img.empty()) {
    //   im_track.setTo(cv::Scalar(0,0,0), mask_img);
    // }

    // gather imu measurements between frames
    // Load imu measurements from previous frame
    vImuMeas.clear();
    while(imuTimestamps[last_imu_idx] <= tframe && tframe > 0)
    {
        vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[last_imu_idx].x,vAcc[last_imu_idx].y,vAcc[last_imu_idx].z,
                                                  vGyr[last_imu_idx].x,vGyr[last_imu_idx].y,vGyr[last_imu_idx].z,
                                                  imuTimestamps[last_imu_idx]));
        last_imu_idx++;
    }

    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();

    // Pass the image to the SLAM system
    // auto result = SLAM.LocalizeMonocular(im_track, tframe, vImuMeas); 
    auto result = SLAM.LocalizeMonocular(im_track, mask_img, tframe, vImuMeas);
    // check lost frames
    if (! result.second){
        n_lost_frames += 1;
        std::cout << "n_lost_frames=" << n_lost_frames << std::endl;
    }
    if ((max_lost_frames > 0) && (n_lost_frames >= max_lost_frames)){
        std::cout << "Lost tracking on " << n_lost_frames << " >= " << max_lost_frames << " frames. Terminating!" << std::endl;
        SLAM.Shutdown();
        return 1;
    }

    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    if (frame_idx % 100 == 0) {
      std::cout<<"Video FPS: "<<fps<<"\n";
      std::cout<<"ORB-SLAM 3 running at: "<<1./ttrack<< " FPS\n";
    }
  }

  // Stop all threads
  SLAM.Shutdown();


  // Save camera trajectory
  if (!output_trajectory_tum.empty()) {
    SLAM.SaveTrajectoryTUM(output_trajectory_tum);
  }

  if (!output_trajectory_csv.empty()) {
    SLAM.SaveTrajectoryCSV(output_trajectory_csv);
  }

  return 0;
}


