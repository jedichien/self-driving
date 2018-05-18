#ifndef PID_H
#define PID_H
#include <vector>
class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  /*
   * Twiddle
   */
  std::vector<double> dp;
  double best_error;
  unsigned int steps;  
  unsigned int op;
  unsigned int param_idx;
  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
  
  /*
   * Calculate twiddling value.
   */
  void Twiddle(double cte);

  /*
   * Twiddle Parameter's operation
   */
  void TwiddleOp(unsigned int param_idx, double value);
};

#endif /* PID_H */
