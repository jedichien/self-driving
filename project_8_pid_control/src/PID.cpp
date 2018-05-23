#include "PID.h"
#include <iostream>
#include <limits>
#include <cmath>

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
  
  this->phase = this->PHASE::INIT;
  this->param_idx = 0;
  this->steps = 0;
  this->best_error = std::numeric_limits<double>::max();
  this->dp = {0.01, 0.0001, 0.1};
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error;
  p_error = cte;
  i_error += cte;
}

double PID::TotalError() {
  return -(Kp*p_error) - (Ki*i_error) - (Kd*d_error);
}
/**
 * There are three phases including INIT, FIRST, SECOND within twiddle.
 * In INIT status, we just init the value to controlling factor.
 * FIRST, we decide whether should give award or punishment for the factor.
 * SECOND, we give final chance for the factor to see it gets improved or not. As a result, if the CTE is not improved, and we set it back to the previous, and then decrease the controlling parameter.
 * 
 */
void PID::Twiddle(double cte) {
  double sum = 0.0;
  for(const auto v : dp) {
    sum += v;
  }
  if(sum < EPS) {
    return;
  }
  
  if(phase == PHASE::INIT) {
    TwiddleOp(param_idx, dp[param_idx]);
    phase = PHASE::FIRST;
  } 
  else if(phase == PHASE::FIRST) {
    if(fabs(cte) < best_error) {
      best_error = fabs(cte);
      dp[param_idx] *= 1.1;
      param_idx = (param_idx+1)%3;
      phase = PHASE::INIT;  
    }
    else {
      TwiddleOp(param_idx, -2*dp[param_idx]);
      phase = PHASE::SECOND;
    }
  }
  else { // SECOND
    if(fabs(cte) < best_error) {
      best_error = fabs(cte);
      dp[param_idx] *= 1.1;
    }
    else {
      TwiddleOp(param_idx, dp[param_idx]);
      dp[param_idx] *= 0.9;
    }
    param_idx = (param_idx+1)%3;
    phase = PHASE::INIT;
  } 
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
