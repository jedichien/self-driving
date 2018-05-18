#include "PID.h"
#include <iostream>
#include <limits>
#include <cmath>

#define FACTOR 0.01
#define EPS 1e-6
#define MAX_STEPS 2000

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->p_error = 0.0;
  this->i_error = 0.0;
  this->d_error = 0.0;
  
  this->op = 0;
  this->param_idx = 0;
  this->steps = 0;
  this->best_error = std::numeric_limits<double>::max();
  this->dp = {FACTOR*Kp, FACTOR*Ki, FACTOR*Kd};
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
}

double PID::TotalError() {
  return -(Kp*p_error) - (Ki*i_error) - (Kd*d_error);
}

void PID::Twiddle(double cte) {
  double sum = 0.0;
  for(const auto v : dp) {
    sum += v;
  }
  if(sum < EPS || steps > MAX_STEPS) {
    steps += 1;
    return;
  }
  if(fabs(cte) < best_error) {
    best_error = fabs(cte);
    dp[param_idx] *= 1.1;
    op = 0;
  }
  else {
    switch(op) {
      case 0:
        op = 1;
        break;
      case 1:
        TwiddleOp(param_idx, -2*dp[param_idx]);
        op = 2;
        break;
      case 2:
        TwiddleOp(param_idx, dp[param_idx]);
        dp[param_idx] *= 0.9;
        op = 0;
        break;
    }
  }
  param_idx = (param_idx+1)%3; 
  steps += 1; 
}

void PID::TwiddleOp(unsigned int param_idx, double value) {
  switch(param_idx) {
    case 0:
      Kp += value;
      break;
    case 1:
      Ki += value;
      break;
    case 2:
      Kd += value;
      break;
  }
}
