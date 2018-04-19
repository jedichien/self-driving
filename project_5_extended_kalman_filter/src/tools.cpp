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
  TODO:
    * Calculate the RMSE here.
  */
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3, 4);

    float px = x_state(0); // position of x
    float py = x_state(1); // position of y
    float vx = x_state(2); // velocity of x
    float vy = x_state(3); // velocity of y

    float c1 = px*px + py*py;
    float c2 = sqrt(c1);
    float c3 = c1*c2;

    // check to avoid division by zero which will cause infinite value.
    if (std::abs(c1) < 0.0001) {
        std::cout << "Error in Calculate Jacobian. Division by zero." << std::endl;
        return Hj;
    }
    // compute Jacobian matrix
    Hj << px/c2, py/c2, 0, -py/c1, 
          px/c1, 0, py*(vx*py - vy // why this equation?

}
