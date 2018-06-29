#ifndef PATH_PLANNING_TRAJECTORY
#define PATH_PLANNING_TRAJECTORY

#include <cmath>
#include <iostream>
#include <vector>

#include "map.h"
#include "behavior.h"
#include "spline.h"
#include "utility.h"
#include "cost.h"
#include "params.h"
#include "prediction.h"
#include "../Eigen-3.3/Eigen/Dense"

struct PointC2 {
  double f;
  double f_dot;
  double f_ddot;
  PointC2(double y=0, double y_dot=0, double y_ddot=0) : f(y), f_dot(y_dot), f_ddot(y_ddot) {}
};

struct TrajectorySD {
  std::vector<PointC2> path_s;
  std::vector<PointC2> path_d;
  TrajectorySD(std::vector<PointC2> S={}, std::vector<PointC2> D={}) : path_s(S), path_d(D) {}
};

struct TrajectoryXY {
  std::vector<double> x_vals;
  std::vector<double> y_vals;
  TrajectoryXY(std::vector<double> X={}, std::vector<double> Y={}) : x_vals(X), y_vals(Y) {}
};

struct TrajectoryJMT {
  TrajectoryXY trajectory;
  TrajectorySD path_sd;
};

struct PreviousPath {
  TrajectoryXY xy;
  TrajectorySD sd;
  int num_xy_reused;
  PreviousPath(TrajectoryXY XY={}, TrajectorySD SD={}, int N=0) : xy(XY), sd(SD), num_xy_reused(N) {}
};

TrajectoryJMT JMT_init(double car_s, double car_d);

class Trajectory {
public:
  Trajectory(std::vector<Target> targets, Map &map, CarData &car, PreviousPath &previous_path, Prediction &prediction);
  ~Trajectory() {}

  double getMinCost() { return _min_cost; }
  double getMinCostIndex() { return _min_cost_index; }
  TrajectoryXY getMinCostTrajectoryXY() { return _trajectories[_min_cost_index]; }
  TrajectorySD getMinCostTrajectorySD() { return _trajectories_sd[_min_cost_index]; }

private:
  std::vector<class Cost> _costs;
  std::vector<TrajectoryXY> _trajectories;
  std::vector<TrajectorySD> _trajectories_sd;
  double _min_cost;
  int _min_cost_index;

  std::vector<double> JMT (std::vector<double> start, std::vector<double> end, double T);
  double polyeval(std::vector<double> c, double t);
  double polyeval_dot(std::vector<double> c, double t);
  double polyeval_ddot(std::vector<double> c, double t);

  TrajectoryXY generate_trajectory(Target target, Map &map, CarData const &car, PreviousPath const &previous_path);
  TrajectoryJMT generate_trajectory_jmt(Target target, Map &map, PreviousPath const &previous_path);
  TrajectoryJMT generate_trajectory_sd(Target target, Map &map, CarData const &car, PreviousPath const &previous_path);
};

#endif // PATH_PLANNING_TRAJECTORY
