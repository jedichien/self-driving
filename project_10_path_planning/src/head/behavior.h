#ifndef PATH_PLANNING_BEHAVIOR
#define PATH_PLANNING_BEHAVIOR

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

#include "params.h"
#include "utility.h"
#include "prediction.h"

struct Target {
  double lane;
  double velocity;
  double time;
  double accel;
  Target(double l=0, double v=0, double t=0, double a=0) : lane(l), velocity(v), time(t), accel(a) {}
};


class Behavior {
  public:
    Behavior(std::vector<std::vector<double>> const &sensor_fusion, CarData car, Prediction const &prediction);
    virtual ~Behavior();
    std::vector<Target> get_targets() const {
      return _targets;
    };
  private:
    std::vector<Target> _targets;
};


#endif // PATH_PLANNING_BEHAVIOR
