#include <iostream>
#include "tools.h"
#include <assert.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse = VectorXd(4);
  rmse << 0, 0, 0, 0;
  assert (estimations.size()!=0);
  assert (ground_truth.size()!=0);
  assert (estimations.size() == ground_truth.size());

  unsigned int size = estimations.size();

  for (unsigned int idx = 0; idx < size; idx++) {
    VectorXd residual = estimations[idx] - ground_truth[idx];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / size;
  rmse = rmse.array().sqrt();
  return rmse;  
}
