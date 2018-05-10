#ifndef NORMALISATION_FOR_LEARNING
#define NORMALISATION_FOR_LEARNING
#include <cmath>
#include <vector>
#define EPS 0.0000001

inline float normpdf(float x, float m, float s) {
  return (1.0/(s*sqrt(2*M_PI))) * exp(-0.5*pow(x-m, 2)/pow(s, 2));
}

inline std::vector<float> normvector(std::vector<float> vc) {
  float sum = 0.0f;
  for(const float value : vc) {
    sum += pow(value, 2);
  }
  if (sum < EPS) {
    sum = EPS;
  }
  for(unsigned int i = 0; i < vc.size(); i++) {
    //vc[i] = vc[i] / (sum/float(vc.size()));
    vc[i] = vc[i] / sqrt(sum);
  }
  return vc;
}

#endif /* NORMALISATION_FOR_LEARNING */
