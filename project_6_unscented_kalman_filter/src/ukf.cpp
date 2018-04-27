#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  Complete the initialization. See ukf.h for other member properties.
  Hint: one or more values initialized above might be wildly off...
  */
  
  // state dimension
  n_x_ = 5;

  // Augmented state dimension
  // L
  n_aug_ = n_x_ + 2; // create 2 * n_aug_ + 1 points
  
  // Number of sigma points
  // 2L+1
  n_sigma_points_ = 2 * n_aug_ + 1;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_points_);
  
  // Sigma points spreading parameter
  float alpha = 0.0001;
  float kappa = 0.0;
  lambda_ = (alpha*alpha)*(n_aug_+kappa)-n_aug_;
  
  weights_ = VectorXd(n_sigma_points_);
  weights_.segment(1, 2*n_aug_).fill(1.0/2*(n_aug_+lambda_));
 
  is_initialized_ = false; 
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  // initial
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);
      float ro_dot = meas_package.raw_measurements_(2);
      
      float px = cos(theta) * ro;
      float py = sin(theta) * ro;
      float vx = cos(theta) * ro_dot;
      float vy = sin(theta) * ro_dot;
      float v = sqrt(vx*vx + vy*vy);
      x_ << px, py, v, 0.0, 0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
    }
    P_ = MatrixXd::Identity(n_x_, n_x_);
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return; 
  }
  /**
   * Prediction
   */
  // F function
  double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);
  /**
   * Update
   */
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else {
    UpdateLidar(meas_package);
  }

}

/**
 * Compute Sigma Points in Prediction state, First step is expanding x_ to x_aug for computing. Second step is augmenting state covariance. Finally, we can use augmented x_ and state covariance to predict sigma points in the next step.  
 * @dt time between k and k+1 in s
 */
void UKF::ComputeSigmaPoints(double dt) {
  // augmented Mean State
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
  Xsig_aug.col(0) = x_aug;
  MatrixXd L = P_aug.llt().matrixL();
  // unscented transformation calculation
  // formula (15)
  for (unsigned int i = 1; i < n_aug_; i++) {
    Xsig_aug.col(i) = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
    Xsig_aug.col(i+n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*L.col(i);
  }
  
  // compute sigma points
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
  
    if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001) {
      p_x = 0.001;
      p_y = 0.001;
    }

    // predicted state values
    double p_xp, p_yp;
    // if yawd too big, we should correct the yaw rotation. see: https://en.wikipedia.org/wiki/Directional_stability   
    if (fabs(yawd) > 0.001) {
      double v_grow = v / yawd;
      p_xp = p_x + v_grow * (sin(yaw + yaw*dt) - sin(yaw));
      p_yp = p_y + v_grow * (cos(yaw) - cos(yaw + yawd * dt));
    } else { // normal case
      double v_grow = v * dt;
      p_xp = p_x + v_grow * cos(yaw);
      p_yp = p_y + v_grow * sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd * dt;
    double yawd_p = yawd;
    
    // add noise to predicted state value
    double dt_2 = dt * dt;
    p_xp += 0.5 * nu_a * dt_2 * cos(yaw);
    p_yp += 0.5 * nu_a * dt_2 * sin(yaw);
    v_p += nu_a * dt;
    yaw_p += 0.5 * nu_yawdd * dt_2;
    yawd_p += nu_yawdd * dt;

    // assign value
    Xsig_pred_(0, i) = p_xp;
    Xsig_pred_(1, i) = p_yp;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Compute Sigma Points
  ComputeSigmaPoints(delta_t);

  // Predicted State Mean
  // TODO
  x_ = Xsig_pred_ * weights_;
  
  // Predicted State Covariance matrix
  P_.fill(0.0);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * prediction
   */
  // sigma points matrix in measurement space
  // transform sigma points into measurement space
  unsigned int n_z = 2;
  // copy sigma points matrix in prediction as measurement matrix. In Eigen the symbol `=` means copy
  // see: https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
  MatrixXd Zsig = Xsig_pred_;
  

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  
  // x_
  // covariance
  // P_
  // NIS
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, unsigned int n_z) {
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // TODO
  z_pred = Zsig * weights_;

  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI; 
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  // add measurement noise to covariance matrix
  // noise matrix of Radar or LIDAR
  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    R = R_radar_;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    R = R_lidar_;
  }
  S = S + R;

  // Cross Covariance Matrix within prediction and measurement
  MatrixXd Tc = MatrixXd(n_x, n_z);
  Tc.fill(0.0);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Measurements
  VectorXd z = meas_package.raw_measurements_;
  // Kalman factor K
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * S.transpose();

  // NIS Calculation
  // TODO
  // why should we do that?
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    NIS_radar = z.transpose() * S.inverse() * z;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    NIS_laser = z.transpose() * S.inverse() * z;
  }

}
