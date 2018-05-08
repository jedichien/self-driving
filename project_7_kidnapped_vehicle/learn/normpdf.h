#ifndef NORMAL_DISTRIBUTION_FOR_LEARNING
#define NORMAL_DISTRIBUTION_FOR_LEARNING
#include <cmath>

inline float normpdf(float x, float m, float s) {
  return (1.0/(s*sqrt(2*M_PI))) * exp(-0.5*pow(x-m, 2)/pow(s, 2));
}

#endif /* NORMAL_DISTRIBUTION_FOR_LEARNING */
