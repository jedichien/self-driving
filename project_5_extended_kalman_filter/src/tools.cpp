#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
     * Root Mean Square Error
     * sqrt( sum( (x-m)^2 )/N )
     */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    // check for should not be zero
    if (estimations.size() == 0) {
        std::cout << "Estimate is empty" << std::endl;
        return rmse;
    }
    // same size check
    if (estimations.size() != ground_truth.size()) {
        std::cout << "Invalid estimations or ground_truth. Data should be the same dimension" << std::endl;
        return rmse;
    }
    // accumulated squared residuals
    for (unsigned int i=0; i < estimations.size(); i++) {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    rmse = rmse/estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3, 4);

    float px = x_state(0); // position of x
    float py = x_state(1); // position of y
    float vx = x_state(2); // velocity of x
    float vy = x_state(3); // velocity of y
    // preclude for specific case
    if (fabs(px) < 0.0001 and fabs(py) < 0.0001) {
        px = 0.0001;
        py = 0.0001;
    }
    float c1 = px*px + py*py;
    // preclude for specific case
    if (fabs(c1) < 0.0000001) {
        c1 = 0.0000001;
    }
    float c2 = sqrt(c1);
    float c3 = c1*c2;

    // check to avoid division by zero which will cause infinite value.
    if (std::abs(c1) < 0.0001) {
        std::cout << "Error in Calculate Jacobian. Division by zero." << std::endl;
        return Hj;
    }
    // compute Jacobian matrix
    // d(c2)/px, d(c2)/py, d(c2)/vx, d(c2)/vy
    // ?
    // ?
    Hj << px/c2, py/c2, 0, 0, 
          -py/c1, px/c1, 0, 0,
          py*(vx*py-vy*px)/c3, px*(px*vy-py*vx)/c3, px/c2, py/c2;
    return Hj;
}
