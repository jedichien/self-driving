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

  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;
  
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
  lambda_ = pow(alpha, 1) * (n_aug_+kappa) - n_aug_;
  //lambda_ = 10 - n_aug_; // why? in the paper, which give above equation instead of this
  
  weights_ = VectorXd(n_sigma_points_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.segment(1, 2*n_aug_).fill(0.5/(n_aug_+lambda_));
  
  weights_c_ = VectorXd(n_sigma_points_);
  weights_c_(0) = lambda_ / (lambda_ + n_aug_) + (1-pow(alpha, 2)+2);
  weights_c_.segment(1, 2*n_aug_).fill(0.5/(n_aug_+lambda_));

  cout << "weights_: \n" << weights_ << endl;
  cout << "weights_c_: \n" << weights_c_ << endl; 
  time_us_ = 0;

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

  /**
   * ==============================================
   *            Initialization
   * ==============================================
   */
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
   * ==============================================
   *            Predict
   * ==============================================
   */
  cout << "===== start prediction ====" << endl; 
  
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  /*
  const double _ddt = 0.05;
  while (dt > 0.1) {
    Prediction(_ddt);
    dt -= _ddt;
  }
  */
  Prediction(dt);
  cout << "------- end prediction -------" << endl;
  /**
   * ==============================================
   *            Update
   * ==============================================
   */
  cout << "===== start update =======" << endl;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }
  cout << "------ end update --------" << endl;

  time_us_ = meas_package.timestamp_;
  return;
}

/**
 * Compute Sigma Points in Prediction state, First step is expanding x_ to x_aug for computing. 
 * Second step is augmenting state covariance. Finally, we can use augmented x_ and state covariance to predict sigma points in the next step.  
 * @dt time between k and k+1 in s
 */
void UKF::ComputeSigmaPoints(double dt) {
  /**
   * ==============================================
   *            Augmented Sigma Points
   * ==============================================
   */
  // augmented Mean State
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
 
  // augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  MatrixXd L = P_aug.llt().matrixL();
  
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
  Xsig_aug.col(0) = x_aug;
  
  // unscented transformation calculation
  // formula (15)
  for (unsigned int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*L.col(i);
  }
  
  /**
   * ==============================================
   *            Predict Sigma Points
   * ==============================================
   */
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
  
    // predicted state values
    double p_xp, p_yp;
    
    // if yawd too big, we should correct the yaw rotation.
    // see: https://en.wikipedia.org/wiki/Directional_stability   
    if (fabs(yawd) > 0.001) {
      double v_grow = v / yawd;
      // problem is here I guess
      // this will influence the covariance of yawd too large
      p_xp = p_x + v_grow * (sin(yaw + yawd*dt) - sin(yaw));
      p_yp = p_y + v_grow * (cos(yaw) - cos(yaw + yawd * dt));
    }
    else { // normal case
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

  cout << "Xsig_pred_: " << Xsig_pred_ << endl;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Compute Predictive Sigma Points
  cout << "Compute Sigma Point." << endl;
  ComputeSigmaPoints(delta_t);

  // Predicted State Mean
  x_.fill(0.0);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  cout << "x_: " << x_ << endl;
  
  // Predicted State Covariance matrix
  P_.fill(0.0);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    P_ = P_ + weights_c_(i) * x_diff * x_diff.transpose();
  }
  cout << "P_: " << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // sigma points matrix in measurement space
  // transform sigma points into measurement space
  unsigned int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_points_);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    Zsig(0, i) = Xsig_pred_(0, i); // px
    Zsig(1, i) = Xsig_pred_(1, i); // py
  }

  UpdateUKF(meas_package, Zsig, n_z);

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
  unsigned int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_points_);
  // transform sigma points into measurement space
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // avoid zero
    if (fabs(p_x) <= 0.0001) {
      p_x = 0.0001;
    }
    if (fabs(p_y) <= 0.0001) {
      p_y = 0.0001;
    }

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    // measurement model, H matrix
    double r = sqrt(p_x*p_x + p_y*p_y);
    double phi = atan2(p_y, p_x);
    double r_dot = (p_x*v1 + p_y*v2) / r;
    Zsig(0, i) = r;
    Zsig(1, i) = phi;
    Zsig(2, i) = r_dot;
  }

  UpdateUKF(meas_package, Zsig, n_z);

}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, unsigned int n_z) {
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  
  // Cross Covariance Matrix within prediction and measurement
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (unsigned int i = 0; i < n_sigma_points_; i++) {
    // measurement covariance matrix S
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI; 
      while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
    }
    S = S + weights_c_(i) * z_diff * z_diff.transpose();
    
    // Cross Covariance Matrix Tc
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;
    } 
    Tc = Tc + weights_c_(i) * x_diff * z_diff.transpose();
  }
  
  // add measurement noise to covariance matrix
  // noise matrix of Radar or LIDAR
  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    R << pow(std_radr_, 2), 0, 0,
         0, pow(std_radphi_, 2), 0,
         0, 0, pow(std_radrd_, 2);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    R << pow(std_laspx_, 2), 0,
         0, pow(std_laspy_, 2);
  }
  S = S + R;
  
  // Measurements update
  VectorXd z = meas_package.raw_measurements_.segment(0, n_z);
  VectorXd z_diff = z - z_pred;
  // angle normalization
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;
  }
 
  // Kalman factor
  MatrixXd K = Tc * S.inverse();

  // Estimate
  x_ = x_ + K * z_diff;
  
  // Error Covariance
  P_ = P_ - K * S * K.transpose();

  cout << "x_: " << x_ << endl;
  cout << "P_: " << P_ << endl;

  // NIS Calculation
  // why should we do that?  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  }
  
}




