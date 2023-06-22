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

// implemented by Steffen Urban, urbste@googlemail.com, 2021

#include "DoubleSphere.h"

#include <boost/serialization/export.hpp>

#include <Eigen/Core>

namespace ORB_SLAM3 {

cv::Point2f DoubleSphere::project(const cv::Point3f &p3D) {

  const double &xi = mvParameters[4];
  const double &alpha = mvParameters[5];

  const double &x = p3D.x;
  const double &y = p3D.y;
  const double &z = p3D.z;

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = sqrt(d1_2);

  const double w1 = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
  const double w2 = (w1 + xi) / sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  if (z <= -w2 * d1) {
    return cv::Point2f(0.f, 0.f);
  }

  const double k = xi * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = sqrt(d2_2);

  const double norm = alpha * d2 + (double(1) - alpha) * k;

  const double mx = x / norm;
  const double my = y / norm;

  return cv::Point2f(mvParameters[0] * mx + mvParameters[2],
                     mvParameters[1] * my + mvParameters[3]);
}

cv::Point2f DoubleSphere::project(const cv::Matx31f &m3D) {
  return this->project(cv::Point3f(m3D(0), m3D(1), m3D(2)));
}

cv::Point2f DoubleSphere::project(const cv::Mat &m3D) {
  const float *p3D = m3D.ptr<float>();

  return this->project(cv::Point3f(p3D[0], p3D[1], p3D[2]));
}

Eigen::Vector2d DoubleSphere::project(const Eigen::Vector3d &v3D) {

  Eigen::Vector2d res;
  res.setZero();
  const double &xi = mvParameters[4];
  const double &alpha = mvParameters[5];

  const double &x = v3D[0];
  const double &y = v3D[1];
  const double &z = v3D[2];

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = sqrt(d1_2);

  const double w1 = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
  const double w2 = (w1 + xi) / sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  if (z <= -w2 * d1) {
    return res;
  }

  const double k = xi * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = sqrt(d2_2);

  const double norm = alpha * d2 + (double(1) - alpha) * k;

  const double mx = x / norm;
  const double my = y / norm;

  res[0] = mvParameters[0] * mx + mvParameters[2];
  res[1] = mvParameters[1] * my + mvParameters[3];

  return res;
}

cv::Mat DoubleSphere::projectMat(const cv::Point3f &p3D) {
  cv::Point2f point = this->project(p3D);
  cv::Mat ret = (cv::Mat_<float>(2, 1) << point.x, point.y);
  return ret.clone();
}

float DoubleSphere::uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D) {
  return 1.f;
}

cv::Mat DoubleSphere::unprojectMat(const cv::Point2f &p2D) {
  cv::Point3f ray = this->unproject(p2D);
  cv::Mat ret = (cv::Mat_<float>(3, 1) << ray.x, ray.y, ray.z);
  return ret.clone();
}

cv::Matx31f DoubleSphere::unprojectMat_(const cv::Point2f &p2D) {
  cv::Point3f ray = this->unproject(p2D);
  cv::Matx31f r{ray.x, ray.y, ray.z};
  return r;
}

cv::Point3f DoubleSphere::unproject(const cv::Point2f &p2D) {

  const double &xi = mvParameters[4];
  const double &alpha = mvParameters[5];

  const double mx = (p2D.x - mvParameters[2]) / mvParameters[0];
  const double my = (p2D.y - mvParameters[3]) / mvParameters[1];

  const double r2 = mx * mx + my * my;

  if (alpha > double(0.5)) {
    if (r2 >= double(1) / (double(2) * alpha - double(1)))
      return cv::Point3f(0.f, 0.f, 1.f);
  }

  const double xi2_2 = alpha * alpha;
  const double xi1_2 = xi * xi;

  const double sqrt2 = sqrt(double(1) - (double(2) * alpha - double(1)) * r2);

  const double norm2 = alpha * sqrt2 + double(1) - alpha;

  const double mz = (double(1) - xi2_2 * r2) / norm2;
  const double mz2 = mz * mz;

  const double norm1 = mz2 + r2;
  const double sqrt1 = sqrt(mz2 + (double(1) - xi1_2) * r2);
  const double k = (mz * xi + sqrt1) / norm1;

  const double x = k * mx;
  const double y = k * my;
  const double z = k * mz - xi;

  return cv::Point3f(x / z, y / z, 1.f);
}

cv::Mat DoubleSphere::projectJac(const cv::Point3f &p3D) {

  cv::Mat Jac = cv::Mat::zeros(2, 3, CV_32F);
  const double &xi = mvParameters[4];
  const double &alpha = mvParameters[5];

  const double &fx = mvParameters[0];
  const double &fy = mvParameters[1];

  const double &x = p3D.x;
  const double &y = p3D.y;
  const double &z = p3D.z;

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = sqrt(d1_2);

  const double w1 = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
  const double w2 = (w1 + xi) / sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  if (z <= -w2 * d1) {
    return Jac;
  }

  const double k = xi * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = sqrt(d2_2);

  const double norm = alpha * d2 + (double(1) - alpha) * k;
  const double norm2 = norm * norm;
  const double xy = x * y;
  const double tt2 = xi * z / d1 + double(1);

  const double d_norm_d_r2 =
      (xi * (double(1) - alpha) / d1 + alpha * (xi * k / d1 + double(1)) / d2) /
      norm2;

  const double tmp2 =
      ((double(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

  Jac.ptr<float>(0)[0] = fx * (double(1) / norm - xx * d_norm_d_r2);
  Jac.ptr<float>(1)[0] = -fy * xy * d_norm_d_r2;

  Jac.ptr<float>(0)[1] = -fx * xy * d_norm_d_r2;
  Jac.ptr<float>(1)[1] = fy * (double(1) / norm - yy * d_norm_d_r2);

  Jac.ptr<float>(0)[2] = -fx * x * tmp2;
  Jac.ptr<float>(1)[2] = -fy * y * tmp2;

  return Jac.clone();
}

Eigen::Matrix<double, 2, 3>
DoubleSphere::projectJac(const Eigen::Vector3d &v3D) {
  Eigen::Matrix<double, 2, 3> JacGood;
  JacGood.setZero();
  const double &xi = mvParameters[4];
  const double &alpha = mvParameters[5];

  const double &fx = mvParameters[0];
  const double &fy = mvParameters[1];

  const double &x = v3D[0];
  const double &y = v3D[1];
  const double &z = v3D[2];

  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;

  const double r2 = xx + yy;

  const double d1_2 = r2 + zz;
  const double d1 = sqrt(d1_2);

  const double w1 = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
  const double w2 = (w1 + xi) / sqrt(2.0 * w1 * xi + xi * xi + 1.0);
  if (z <= -w2 * d1) {
    return JacGood;
  }

  const double k = xi * d1 + z;
  const double kk = k * k;

  const double d2_2 = r2 + kk;
  const double d2 = sqrt(d2_2);

  const double norm = alpha * d2 + (double(1) - alpha) * k;
  const double norm2 = norm * norm;
  const double xy = x * y;
  const double tt2 = xi * z / d1 + double(1);

  const double d_norm_d_r2 =
      (xi * (double(1) - alpha) / d1 + alpha * (xi * k / d1 + double(1)) / d2) /
      norm2;

  const double tmp2 =
      ((double(1) - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

  JacGood(0, 0) = fx * (double(1) / norm - xx * d_norm_d_r2);
  JacGood(1, 0) = -fy * xy * d_norm_d_r2;

  JacGood(0, 1) = -fx * xy * d_norm_d_r2;
  JacGood(1, 1) = fy * (double(1) / norm - yy * d_norm_d_r2);

  JacGood(0, 2) = -fx * x * tmp2;
  JacGood(1, 2) = -fy * y * tmp2;

  return JacGood;
}

cv::Mat DoubleSphere::unprojectJac(const cv::Point2f &p2D) { return cv::Mat(); }

bool DoubleSphere::ReconstructWithTwoViews(
    const std::vector<cv::KeyPoint> &vKeys1,
    const std::vector<cv::KeyPoint> &vKeys2, const std::vector<int> &vMatches12,
    cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D,
    std::vector<bool> &vbTriangulated) {
  if (!tvr) {
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);// this->toK();
    tvr = new TwoViewReconstruction(K);
  }

  // Correct FishEye distortion
  std::vector<cv::KeyPoint> vKeysUn1(vKeys1.size()), vKeysUn2(vKeys2.size());
  for (size_t i = 0; i < vKeys1.size(); i++) {
    const cv::Point3f pt3 = this->unproject(vKeys1[i].pt);
    vKeysUn1[i].pt.x = pt3.x / pt3.z;
    vKeysUn1[i].pt.y = pt3.y / pt3.z;
  }
  for (size_t i = 0; i < vKeys2.size(); i++) {
    const cv::Point3f pt3 = this->unproject(vKeys2[i].pt);
    vKeysUn2[i].pt.x = pt3.x / pt3.z;
    vKeysUn2[i].pt.y = pt3.y / pt3.z;
  }
  return tvr->Reconstruct(vKeysUn1, vKeysUn2, vMatches12, R21, t21, vP3D,
                          vbTriangulated);
}

cv::Mat DoubleSphere::toK() {
  cv::Mat K = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2],
               0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
  return K;
}

cv::Matx33f DoubleSphere::toK_() {
  cv::Matx33f K{mvParameters[0],
                0.f,
                mvParameters[2],
                0.f,
                mvParameters[1],
                mvParameters[3],
                0.f,
                0.f,
                1.f};

  return K;
}

bool DoubleSphere::epipolarConstrain(GeometricCamera *pCamera2,
                                     const cv::KeyPoint &kp1,
                                     const cv::KeyPoint &kp2,
                                     const cv::Mat &R12, const cv::Mat &t12,
                                     const float sigmaLevel, const float unc) {
  cv::Mat p3D;
  return this->TriangulateMatches(pCamera2, kp1, kp2, R12, t12, sigmaLevel, unc,
                                  p3D) > 0.0001f;
}

bool DoubleSphere::epipolarConstrain_(GeometricCamera *pCamera2,
                                      const cv::KeyPoint &kp1,
                                      const cv::KeyPoint &kp2,
                                      const cv::Matx33f &R12,
                                      const cv::Matx31f &t12,
                                      const float sigmaLevel, const float unc) {
  cv::Matx31f p3D;
  return this->TriangulateMatches_(pCamera2, kp1, kp2, R12, t12, sigmaLevel,
                                   unc, p3D) > 0.0001f;
}

bool DoubleSphere::matchAndtriangulate(const cv::KeyPoint &kp1,
                                       const cv::KeyPoint &kp2,
                                       GeometricCamera *pOther, cv::Mat &Tcw1,
                                       cv::Mat &Tcw2, const float sigmaLevel1,
                                       const float sigmaLevel2,
                                       cv::Mat &x3Dtriangulated) {
  cv::Mat Rcw1 = Tcw1.colRange(0, 3).rowRange(0, 3);
  cv::Mat Rwc1 = Rcw1.t();
  cv::Mat tcw1 = Tcw1.rowRange(0, 3).col(3);

  cv::Mat Rcw2 = Tcw2.colRange(0, 3).rowRange(0, 3);
  cv::Mat Rwc2 = Rcw2.t();
  cv::Mat tcw2 = Tcw2.rowRange(0, 3).col(3);

  cv::Point3f ray1c = this->unproject(kp1.pt);
  cv::Point3f ray2c = pOther->unproject(kp2.pt);

  cv::Mat r1(3, 1, CV_32F);
  r1.at<float>(0) = ray1c.x;
  r1.at<float>(1) = ray1c.y;
  r1.at<float>(2) = ray1c.z;

  cv::Mat r2(3, 1, CV_32F);
  r2.at<float>(0) = ray2c.x;
  r2.at<float>(1) = ray2c.y;
  r2.at<float>(2) = ray2c.z;

  // Check parallax between rays
  cv::Mat ray1 = Rwc1 * r1;
  cv::Mat ray2 = Rwc2 * r2;

  const float cosParallaxRays =
      ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

  // If parallax is lower than 0.9998, reject this match
  if (cosParallaxRays > 0.9998) {
    return false;
  }

  // Parallax is good, so we try to triangulate
  cv::Point2f p11, p22;

  p11.x = ray1c.x;
  p11.y = ray1c.y;

  p22.x = ray2c.x;
  p22.y = ray2c.y;

  cv::Mat x3D;

  Triangulate(p11, p22, Tcw1, Tcw2, x3D);

  cv::Mat x3Dt = x3D.t();

  // Check triangulation in front of cameras
  float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
  if (z1 <= 0) { // Point is not in front of the first camera
    return false;
  }

  float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
  if (z2 <= 0) { // Point is not in front of the first camera
    return false;
  }

  // Check reprojection error in first keyframe
  //  -Transform point into camera reference system
  cv::Mat x3D1 = Rcw1 * x3D + tcw1;
  cv::Point2f uv1 = this->project(x3D1);

  float errX1 = uv1.x - kp1.pt.x;
  float errY1 = uv1.y - kp1.pt.y;

  if ((errX1 * errX1 + errY1 * errY1) >
      5.991 * sigmaLevel1) { // Reprojection error is high
    return false;
  }

  // Check reprojection error in second keyframe;
  //  -Transform point into camera reference system
  cv::Mat x3D2 = Rcw2 * x3D + tcw2;
  cv::Point2f uv2 = pOther->project(x3D2);

  float errX2 = uv2.x - kp2.pt.x;
  float errY2 = uv2.y - kp2.pt.y;

  if ((errX2 * errX2 + errY2 * errY2) >
      5.991 * sigmaLevel2) { // Reprojection error is high
    return false;
  }

  // Since parallax is big enough and reprojection errors are low, this pair of
  // points can be considered as a match
  x3Dtriangulated = x3D.clone();

  return true;
}

float DoubleSphere::TriangulateMatches(GeometricCamera *pCamera2,
                                       const cv::KeyPoint &kp1,
                                       const cv::KeyPoint &kp2,
                                       const cv::Mat &R12, const cv::Mat &t12,
                                       const float sigmaLevel, const float unc,
                                       cv::Mat &p3D) {
  cv::Mat r1 = this->unprojectMat(kp1.pt);
  cv::Mat r2 = pCamera2->unprojectMat(kp2.pt);

  // Check parallax
  cv::Mat r21 = R12 * r2;

  const float cosParallaxRays = r1.dot(r21) / (cv::norm(r1) * cv::norm(r21));

  if (cosParallaxRays > 0.9998) {
    return -1;
  }

  // Parallax is good, so we try to triangulate
  cv::Point2f p11, p22;
  const float *pr1 = r1.ptr<float>();
  const float *pr2 = r2.ptr<float>();

  p11.x = pr1[0];
  p11.y = pr1[1];

  p22.x = pr2[0];
  p22.y = pr2[1];

  cv::Mat x3D;
  cv::Mat Tcw1 = (cv::Mat_<float>(3, 4) << 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
                  0.f, 0.f, 0.f, 1.f, 0.f);
  cv::Mat Tcw2;
  cv::Mat R21 = R12.t();
  cv::Mat t21 = -R21 * t12;
  cv::hconcat(R21, t21, Tcw2);

  Triangulate(p11, p22, Tcw1, Tcw2, x3D);
  cv::Mat x3Dt = x3D.t();

  float z1 = x3D.at<float>(2);
  if (z1 <= 0) {
    return -1;
  }

  float z2 = R21.row(2).dot(x3Dt) + t21.at<float>(2);
  if (z2 <= 0) {
    return -1;
  }

  // Check reprojection error
  cv::Point2f uv1 = this->project(x3D);

  float errX1 = uv1.x - kp1.pt.x;
  float errY1 = uv1.y - kp1.pt.y;

  if ((errX1 * errX1 + errY1 * errY1) >
      5.991 * sigmaLevel) { // Reprojection error is high
    return -1;
  }

  cv::Mat x3D2 = R21 * x3D + t21;
  cv::Point2f uv2 = pCamera2->project(x3D2);

  float errX2 = uv2.x - kp2.pt.x;
  float errY2 = uv2.y - kp2.pt.y;

  if ((errX2 * errX2 + errY2 * errY2) >
      5.991 * unc) { // Reprojection error is high
    return -1;
  }

  p3D = x3D.clone();

  return z1;
}

float DoubleSphere::TriangulateMatches_(
    GeometricCamera *pCamera2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
    const cv::Matx33f &R12, const cv::Matx31f &t12, const float sigmaLevel,
    const float unc, cv::Matx31f &p3D) {
  cv::Matx31f r1 = this->unprojectMat_(kp1.pt);
  cv::Matx31f r2 = pCamera2->unprojectMat_(kp2.pt);

  // Check parallax
  cv::Matx31f r21 = R12 * r2;

  const float cosParallaxRays = r1.dot(r21) / (cv::norm(r1) * cv::norm(r21));

  if (cosParallaxRays > 0.9998) {
    return -1;
  }

  // Parallax is good, so we try to triangulate
  cv::Point2f p11, p22;

  p11.x = r1(0);
  p11.y = r1(1);

  p22.x = r2(0);
  p22.y = r2(1);

  cv::Matx31f x3D;
  cv::Matx44f Tcw1{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f};

  cv::Matx33f R21 = R12.t();
  cv::Matx31f t21 = -R21 * t12;

  cv::Matx44f Tcw2{R21(0, 0), R21(0, 1), R21(0, 2), t21(0),
                   R21(1, 0), R21(1, 1), R21(1, 2), t21(1),
                   R21(2, 0), R21(2, 1), R21(2, 2), t21(2),
                   0.f,       0.f,       0.f,       1.f};

  Triangulate_(p11, p22, Tcw1, Tcw2, x3D);
  cv::Matx13f x3Dt = x3D.t();

  float z1 = x3D(2);
  if (z1 <= 0) {
    return -1;
  }

  float z2 = R21.row(2).dot(x3Dt) + t21(2);
  if (z2 <= 0) {
    return -1;
  }

  // Check reprojection error
  cv::Point2f uv1 = this->project(x3D);

  float errX1 = uv1.x - kp1.pt.x;
  float errY1 = uv1.y - kp1.pt.y;

  if ((errX1 * errX1 + errY1 * errY1) >
      5.991 * sigmaLevel) { // Reprojection error is high
    return -1;
  }

  cv::Matx31f x3D2 = R21 * x3D + t21;
  cv::Point2f uv2 = pCamera2->project(x3D2);

  float errX2 = uv2.x - kp2.pt.x;
  float errY2 = uv2.y - kp2.pt.y;

  if ((errX2 * errX2 + errY2 * errY2) >
      5.991 * unc) { // Reprojection error is high
    return -1;
  }

  p3D = x3D;

  return z1;
}

std::ostream &operator<<(std::ostream &os, const DoubleSphere &kb) {
  os << kb.mvParameters[0] << " " << kb.mvParameters[1] << " "
     << kb.mvParameters[2] << " " << kb.mvParameters[3] << " "
     << kb.mvParameters[4] << " " << kb.mvParameters[5];
  return os;
}

std::istream &operator>>(std::istream &is, DoubleSphere &kb) {
  float nextParam;
  for (int i = 0; i < kb.NUM_PARAMS_; i++) {
    assert(is.good()); // Make sure the input stream is good
    is >> nextParam;
    kb.mvParameters[i] = nextParam;
  }
  return is;
}

void DoubleSphere::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2,
                               const cv::Mat &Tcw1, const cv::Mat &Tcw2,
                               cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_32F);

  A.row(0) = p1.x * Tcw1.row(2) - Tcw1.row(0);
  A.row(1) = p1.y * Tcw1.row(2) - Tcw1.row(1);
  A.row(2) = p2.x * Tcw2.row(2) - Tcw2.row(0);
  A.row(3) = p2.y * Tcw2.row(2) - Tcw2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void DoubleSphere::Triangulate_(const cv::Point2f &p1, const cv::Point2f &p2,
                                const cv::Matx44f &Tcw1,
                                const cv::Matx44f &Tcw2, cv::Matx31f &x3D) {
  cv::Matx14f A0, A1, A2, A3;

  A0 = p1.x * Tcw1.row(2) - Tcw1.row(0);
  A1 = p1.y * Tcw1.row(2) - Tcw1.row(1);
  A2 = p2.x * Tcw2.row(2) - Tcw2.row(0);
  A3 = p2.y * Tcw2.row(2) - Tcw2.row(1);
  cv::Matx44f A{A0(0), A0(1), A0(2), A0(3), A1(0), A1(1), A1(2), A1(3),
                A2(0), A2(1), A2(2), A2(3), A3(0), A3(1), A3(2), A3(3)};

  // cv::Mat u,w,vt;
  cv::Matx44f u, vt;
  cv::Matx41f w;

  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  cv::Matx41f x3D_h = vt.row(3).t();
  x3D = x3D_h.get_minor<3, 1>(0, 0) * (1 / x3D_h(3));
}
} // namespace ORB_SLAM3
