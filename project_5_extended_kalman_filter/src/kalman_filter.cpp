#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

/**
 * Taking a briefly introduction about Kalman-Filter.
 * 
 * Kalman-filter aims to get more accuracy of signal measurement.
 * 
 * Because, signal always has noise and others factors, so we have to comes up with the solution.
 * 
 * Therefore, Kalman-filter solves this problem with more flexible method, it provides two way, Prediction and Updating.
 * 
 * I suppose having a sensor for measuring signal in current environment which suffers from noise. However, I have to estimate the more realistic value.
 * 
 *
 * Kalman-Filter can offer me to do that.
 * 
 * First of all, Kalman Filter predicts current value which is not including measurement, only prediction.
 * 
 * Secondly, Kalman uses previous covariance to get the factor, K, for updating previous prediction. In addtion, in this part, we can previous estimate, K factor, and the error value between prediction and measurement to estimate current signal value.
 * 
 * Therefore, we repeat these steps until getting zero covariance or specific times. 
 */
KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /* 
   * Suppose this environment is steable.
   */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  KF(y);
}

// EKF
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */
  float px = x_(0); // x coordinates
  float py = x_(1); // y coordinates
  float vx = x_(2); // velocity of x-axis
  float vy = x_(3); // velocity of y-axis
  
  // distance
  double rho = sqrt(px*px + py*py);
  
  // angle
  double phi = atan2(py, px);

  // velocity vector
  // px dot with vx, and prevent from infinity occuring
  // calculate vector according to measuring velocity.
  double rho_dot = (px*vx + py*vy) / std::max(rho, 0.000001);

  VectorXd h(3);
  // distance, angle, velocity vector
  h << rho, phi, rho_dot;

  VectorXd y = z - h;
  KF(y);  
}

void KalmanFilter::KF(const VectorXd &y) {
  // update state by state by using KF
  MatrixXd H_T = H_.transpose();
  MatrixXd K = P_ * H_T * (H_*P_*H_T + R_).inverse();
  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I-K*H_) * P_;
}
