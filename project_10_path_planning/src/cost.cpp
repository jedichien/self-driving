#include "head/cost.h"

using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::VectorXd;
using Eigen::Vector2d;

bool Cost::check_collision(double x0, double y0, double theta0, double x1, double y1, double theta1) {
  Vector2d trans0;
  trans0 << x0, y0;
  
  Vector2d trans1;
  trans1 << x1, y1;

  Matrix2d rot0, rot1;

  rot0 << std::cos(theta0), -1*std::sin(theta0),
       std::sin(theta0), std::cos(theta0);
  rot1 << std::cos(theta1), -1*std::sin(theta1),
       std::sin(theta1), std::cos(theta1);

  double W = PARAM_CAR_SAFETY_W;
  double L = PARAM_CAR_SAFETY_L;

  MatrixXd car(2, 4);
  car << -L/2, L/2, L/2, -L/2,
      W/2, W/2, -W/2, -W/2;

  MatrixXd car0(2, 4);
  MatrixXd car1(2, 4);

  for (size_t i = 0; i < car.cols(); i++) {
    car0.col(i) = rot0 * car.col(i) + trans0;
    car1.col(i) = rot1 * car.col(i) + trans1;
  }
  // principal axis list
  MatrixXd axis(2, 4);
  axis << std::cos(theta0), -1*std::sin(theta0), std::cos(theta1), -1*std::sin(theta1),
       std::sin(theta0), std::cos(theta0), std::sin(theta1), std::cos(theta1);

  for (size_t i = 0; i < axis.cols(); i++) {
    Vector2d principal_axis = axis.col(i);
    // projection of car0
    double min0 = principal_axis.dot(car0.col(0));
    double max0 = min0;
    for (size_t j = 0; j < car0.cols(); j++) {
      double proj0 = principal_axis.dot(car0.col(j));
      if (proj0 > max0) {
        max0 = proj0;
      }
      if (proj0 < min0) {
        min0 = proj0;
      }
    }

    double min1 = principal_axis.dot(car1.col(0));
    double max1 = min1;
    for (size_t j = 0; j < car1.cols(); j++) {
      double proj1 = principal_axis.dot(car1.col(j));
      if (proj1 > max1) {
        max1 = proj1;
      }
      if (proj1 < min1) {
        min1 = proj1;
      }
    }

    bool overlap = false;
    if (min1 >= min0 && min1 < max0) {
      overlap = true;
    }
    if (max1 >= min0 && max1 < max0) {
      overlap = true;
    }
    if (min0 >= min1 && min0 < max1) {
      overlap = true;
    }
    if (max0 >= min1 && max0 < max1) {
      overlap = true;
    }
    if (!overlap) {
      return false;
    }
  }
  return true;
}

int Cost::check_collision_on_trajectory(TrajectoryXY const &trajectory, std::map<int, std::vector<Coord>> &prediction) {
  std::map<int, std::vector<Coord>>::iterator it = prediction.begin();
  
  while (it != prediction.end()) {
    int fusion_index = it->first;
    std::vector<Coord> pd = it->second;
    
    assert(pd.size() == trajectory.x_vals.size());
    assert(pd.size() == trajectory.y_vals.size());

    for (size_t i = 0; i < PARAM_MAX_COLLISION_STEP; i++) {
      double obj_x = pd[i].x;
      double obj_y = pd[i].y;
      double obj_x_next = pd[i+1].x;
      double obj_y_next = pd[i+1].y;
      double obj_heading = std::atan2(obj_y_next - obj_y, obj_x_next - obj_x);

      double ego_x = trajectory.x_vals[i];
      double ego_y = trajectory.y_vals[i];
      double ego_x_next = trajectory.x_vals[i+1];
      double ego_y_next = trajectory.y_vals[i+1];
      double ego_heading = std::atan2(ego_y_next - ego_y, ego_x_next - ego_x);

      if (check_collision(obj_x, obj_y, obj_heading, ego_x, ego_y, ego_heading)) {
        std::cout << "########### COLLISION PREDICTED ON CANDIDATE TRAJECTORY AT STEP " << i << " ###########" << std::endl;
        return i+1;
      }
    }
    it++;
  }
  return 0;
}

// TODO
bool Cost::check_max_capabilities(std::vector<std::vector<double>> &traj) {
  double vx, ax, jx;
  double vy, ay, jy;
  double vel, acc, jerk;
  double max_vel = 0.0;
  double max_acc = 0.0;
  double total_jerk = 0.0;
  double x, x1, x2, x3;
  double y, y1, y2, y3;
  double jerk_per_second;

  assert(traj[0].size() == traj[1].size());

  for (size_t t = 3; t < traj[0].size(); t++) {
    x = traj[0][t];
    x1 = traj[0][t-1];
    x2 = traj[0][t-2];
    x3 = traj[0][t-3];

    y = traj[1][t];
    y1 = traj[1][t-1];
    y2 = traj[1][t-2];
    y3 = traj[1][t-3];

    vx = (x - x1)/PARAM_DT;
    vy = (y - y1)/PARAM_DT;

    ax = (x - 2*x1 + x2) / pow(PARAM_DT, 2);
    ay = (y - 2*y1 + y2) / pow(PARAM_DT, 2);

    // rounding to 2 decimals(cm unit)
    jx = x - 3*x1 + 3*x2 - x3;
    jx = roundf(jx*100)/100;
    jx = jx / pow(PARAM_DT, 3);

    jy = y - 3*y1 + 3*y2 - y3;
    jy = roundf(jy*100)/100;
    jy = jy / pow(PARAM_DT, 3);

    vel = sqrt(pow(vx, 2) + pow(vy, 2));
    acc = sqrt(pow(ax, 2) + pow(ay, 2));
    jerk = sqrt(pow(jx, 2) + pow(jy, 2));

    total_jerk += jerk * PARAM_DT;

    if (vel > max_vel) {
      max_vel = vel;
    }
    if (acc > max_acc) {
      max_acc = acc;
    }
  }
  jerk_per_second = total_jerk / (PARAM_NB_POINTS * PARAM_DT);

  if (roundf(max_vel) > PARAM_MAX_SPEED || roundf(max_acc) > PARAM_MAX_ACCEL || jerk_per_second > PARAM_MAX_JERK) {
    std::cout << "max_vel=" << max_vel << ", max_acc=" << max_acc << ", jerk_per_second=" << jerk_per_second << std::endl;
    return true;
  }
  return false;
}

// TODO
double Cost::get_predicted_dmin(TrajectoryXY const &trajectory, std::map<int, std::vector<Coord>> &prediction) {
  double dmin = INF;
  std::map<int, std::vector<Coord>>::iterator it = prediction.begin();

  while (it != prediction.end()) {
    int fusion_index = it->first;
    std::vector<Coord> pd = it->second;

    assert(pd.size() == trajectory.x_vals.size());
    assert(pd.size() == trajectory.y_vals.size());

    for (size_t i = 0; i < pd.size(); i++) {
      double obj_x = pd[i].x;
      double obj_y = pd[i].y;
      double ego_x = trajectory.x_vals[i];
      double ego_y = trajectory.y_vals[i];
      double dist = distance(ego_x, ego_y, obj_x, obj_y);

      if (dist < dmin) {
        dmin = dist;
      }
    }
    it++;
  }
  std::cout << "dmin=" << dmin << std::endl;
  return dmin;
}

// TODO
Cost::Cost(TrajectoryXY const &trajectory, Target target, Prediction &prediction, int car_lane) {
  _cost = 0;
  double cost_feasibility = 0.0;
  double cost_safety = 0.0;
  double cost_legality = 0.0;
  double cost_comfort = 0.0;
  double cost_efficiency = 0.0;

  std::map<int, std::vector<Coord>> pd = prediction.get_predictions();
  // Feasibility
  cost_feasibility += check_collision_on_trajectory(trajectory, pd);
  _cost += PARAM_COST_FEASIBILITY * cost_feasibility;
 // Safety(Ignored)
 _cost += PARAM_COST_SAFETY * cost_safety;
 // Legality
 _cost += PARAM_COST_LEGALITY * cost_legality;
 // Comfort
 _cost += PARAM_COST_COMFORT * cost_comfort;
 // Efficiency
 cost_efficiency = PARAM_FOV - prediction.get_lane_free_space(target.lane);
 _cost += PARAM_COST_EFFICIENCY * cost_efficiency;

 std::cout << "car_lane=" << car_lane << ", target_lane=" << target.lane << ", target_lvel=" << prediction.get_lane_speed(target.lane) << ", cost=" << _cost << std::endl;
}
