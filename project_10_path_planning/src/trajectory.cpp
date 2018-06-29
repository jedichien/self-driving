#include "head/trajectory.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// initialize the JMT
TrajectoryJMT JMT_init(double car_s, double car_d) {
  TrajectoryJMT traj_jmt;
  std::vector<PointC2> store_path_s(PARAM_NB_POINTS, PointC2(0, 0, 0));
  std::vector<PointC2> store_path_d(PARAM_NB_POINTS, PointC2(0, 0, 0));
  
  for (size_t i = 0; i < PARAM_NB_POINTS; i++) {
    store_path_s[i] = PointC2(car_s, 0, 0);
    store_path_d[i] = PointC2(car_d, 0, 0);
  }

  traj_jmt.path_sd.path_s = store_path_s;
  traj_jmt.path_sd.path_d = store_path_d;
  
  return traj_jmt;
}

/**
 * TODO
 *
 */
Trajectory::Trajectory(std::vector<Target> targets, Map &map, CarData &car, PreviousPath &previous_path, Prediction &prediction) {
  for (size_t i = 0; i < targets.size(); i++) {
    TrajectoryXY trajectory;
    if (PARAM_TRAJECTORY_JMT) {
      TrajectoryJMT traj_jmt;
      if (targets[i].time == 0) { // tackle emergency
        traj_jmt = generate_trajectory_sd(targets[i], map, car, previous_path);
      }
      else {
        traj_jmt = generate_trajectory_jmt(targets[i], map, previous_path);
      }
      trajectory = traj_jmt.trajectory;
      _trajectories_sd.push_back(traj_jmt.path_sd);
    }
    else {
      trajectory = generate_trajectory(targets[i], map, car, previous_path);
    }
    Cost cost = Cost(trajectory, targets[i], prediction, car.lane);
    _costs.push_back(cost);
    _trajectories.push_back(trajectory);
  }
  // retrieve the lowest trajectory
  _min_cost = INF;
  _min_cost_index = 0;
  
  for (size_t i = 0; i < _costs.size(); i++) {
    if (_costs[i].get_cost() < _min_cost) {
      _min_cost = _costs[i].get_cost();
      _min_cost_index = i;
    }
  }
  
  if (_min_cost >= PARAM_COST_FEASIBILITY) {
    _min_cost_index = _costs.size() - 1;
    _min_cost = _costs[_min_cost_index].get_cost();
  }

  car.emergency = (targets[_min_cost_index].time == 0) ? true : false;

  if (car.emergency) {
    std::cout << "###################### EMERGENCY!!!!!!! ######################" << std::endl;
  }
}

/**
 * Calculate the minimizing jerk trajectory that connects the initial state to the final state in time T.
 */
std::vector<double> Trajectory::JMT(std::vector<double> start, std::vector<double> end, double T) {
  MatrixXd A(3, 3);
  VectorXd b(3);
  VectorXd x(3);

  A << pow(T, 3), pow(T, 4), pow(T, 5),
    3*pow(T, 2), 4*pow(T, 3), 5*pow(T, 4),
    6*T, 12*pow(T, 2), 20*pow(T, 3);

  b << end[0] - (start[0] + start[1]*T + 0.5*start[2]*pow(T, 2)),
    end[1] - (start[1] + start[2]*T),
    end[2] - start[2];

  x = A.inverse() * b;

  return { start[0], start[1], start[2]/2, x[0], x[1], x[2] };
}

double Trajectory::polyeval(std::vector<double> c, double t) {
  double res = 0.0;
  for (size_t i = 0; i < c.size(); i++) {
    res += c[i] * pow(t, i);
  }
  return res;
}

double Trajectory::polyeval_dot(std::vector<double> c, double t) {
  double res = 0.0;
  for (size_t i = 1; i < c.size(); i++) {
    res += i * c[i] * pow(t, i-1);
  }
  return res;
}

double Trajectory::polyeval_ddot(std::vector<double> c, double t) {
  double res = 0.0;
  for (size_t i = 2; i < c.size(); i++) {
    res += i * (i-1) * c[i] * pow(t, i-2);
  }
  return res;
}
/**
 * TODO
 *
 */
TrajectoryJMT Trajectory::generate_trajectory_jmt(Target target, Map &map, PreviousPath const &previous_path) {
  TrajectoryJMT traj_jmt;

  TrajectoryXY previous_path_xy = previous_path.xy;
  int prev_size = previous_path.num_xy_reused;
  TrajectorySD prev_path_sd = previous_path.sd;

  std::vector<double> previous_path_x = previous_path_xy.x_vals;
  std::vector<double> previous_path_y = previous_path_xy.y_vals;
  std::vector<PointC2> prev_path_s = prev_path_sd.path_s;
  std::vector<PointC2> prev_path_d = prev_path_sd.path_d;

  std::vector<PointC2> new_path_s(PARAM_NB_POINTS, PointC2(0, 0, 0));
  std::vector<PointC2> new_path_d(PARAM_NB_POINTS, PointC2(0, 0, 0));

  int last_point;
  if (PARAM_PREV_PATH_XY_REUSED < PARAM_NB_POINTS) {
    last_point = PARAM_NB_POINTS - previous_path_x.size() + prev_size - 1;
  }
  else {
    last_point = PARAM_NB_POINTS - 1;
  }

  double T = target.time;
  double si, si_dot=0, si_ddot;
  double di, di_dot, di_ddot;

  si = prev_path_s[last_point].f;
  si_dot = prev_path_s[last_point].f_dot;
  si_ddot = prev_path_s[last_point].f_ddot;

  di = prev_path_d[last_point].f;
  di_dot = prev_path_d[last_point].f_dot;
  di_ddot = prev_path_d[last_point].f_ddot;

  double sf, sf_dot, sf_ddot;
  double df, df_dot, df_ddot;

  if (target.velocity <= 10) {
    df_ddot = 0;
    df_dot = 0;
    df = di;

    sf_ddot = 0;
    sf_dot = mph_to_ms(target.velocity);
    sf_dot = std::min(sf_dot, si_dot + 10*PARAM_MAX_SPEED_INC);
    sf_dot = std::max(sf_dot, si_dot - 10*PARAM_MAX_SPEED_INC);
    sf = si + 2 * sf_dot * T;
  }
  else {
    df_ddot = 0;
    df_dot = 0;
    df = get_dcenter(target.lane);

    sf_ddot = 0;
    sf_dot = mph_to_ms(target.velocity);
    sf_dot = std::min(sf_dot, 0.9*PARAM_MAX_SPEED);
    sf_dot = std::min(sf_dot, si_dot + 10*PARAM_MAX_SPEED_INC);
    sf_dot = std::max(sf_dot, si_dot - 10*PARAM_MAX_SPEED_INC);

    sf = si + sf_dot * T;
  }
  std::vector<double> start_s = { si, si_dot, si_ddot };
  std::vector<double> end_s = { sf, sf_dot, sf_ddot };

  std::vector<double> start_d = { di, di_dot, di_ddot };
  std::vector<double> end_d = { df, df_dot, df_ddot };

  std::vector<double> poly_s = JMT(start_s, end_s, T);
  std::vector<double> poly_d = JMT(start_d, end_d, T);

  std::vector<double> next_x_vals;
  std::vector<double> next_y_vals;

  for (size_t i = 0; i < prev_size; i++) {
    new_path_s[i] = prev_path_s[PARAM_NB_POINTS - previous_path_x.size() + i];
    new_path_d[i] = prev_path_d[PARAM_NB_POINTS - previous_path_x.size() + i];

    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  double t = PARAM_DT;
  for (size_t i = prev_size; i < PARAM_NB_POINTS; i++) {
    double s = polyeval(poly_s, t);
    double s_dot = polyeval_dot(poly_s, t);
    double s_ddot = polyeval_ddot(poly_s, t);

    double d = polyeval(poly_d, t);
    double d_dot = polyeval_dot(poly_d, t);
    double d_ddot = polyeval_ddot(poly_d, t);

    new_path_s[i] = PointC2(s, s_dot, s_ddot);
    new_path_d[i] = PointC2(d, d_dot, d_ddot);

    std::vector<double> point_xy = map.getXYspline(s, d);

    next_x_vals.push_back(point_xy[0]);
    next_y_vals.push_back(point_xy[1]);

    t += PARAM_DT;
  }
  traj_jmt.trajectory = TrajectoryXY(next_x_vals, next_y_vals);
  traj_jmt.path_sd = TrajectorySD(new_path_s, new_path_d);

  return traj_jmt;
}

/**
 * TODO
 *
 */
TrajectoryJMT Trajectory::generate_trajectory_sd(Target target, Map &map, CarData const &car, PreviousPath const &previous_path) {
  TrajectoryJMT traj_jmt;
  TrajectoryXY prev_path_xy = previous_path.xy;
  int prev_size = previous_path.num_xy_reused;
  TrajectorySD prev_path_sd = previous_path.sd;

  std::vector<double> prev_path_x = prev_path_xy.x_vals;
  std::vector<double> prev_path_y = prev_path_xy.y_vals;
  std::vector<PointC2> prev_path_s = prev_path_sd.path_s;
  std::vector<PointC2> prev_path_d = prev_path_sd.path_d;

  std::vector<PointC2> new_path_s(PARAM_NB_POINTS, PointC2(0, 0, 0));
  std::vector<PointC2> new_path_d(PARAM_NB_POINTS, PointC2(0, 0, 0));

  std::vector<double> next_x_vals;
  std::vector<double> next_y_vals;

  double target_velocity_ms = mph_to_ms(target.velocity);

  double s, s_dot, s_ddot;
  double d, d_dot, d_ddot;

  if (prev_size > 0) {
    for (size_t i = 0; i < prev_size; i++) {
      new_path_s[i] = prev_path_s[PARAM_NB_POINTS - prev_path_x.size() + i];
      new_path_d[i] = prev_path_d[PARAM_NB_POINTS - prev_path_x.size() + i];

      next_x_vals.push_back(prev_path_x[i]);
      next_y_vals.push_back(prev_path_y[i]);
    }
    s = new_path_s[prev_size-1].f;
    s_dot = new_path_s[prev_size-1].f_dot;
    d = new_path_d[prev_size-1].f;
    d_dot = 0;
    d_ddot = 0;
  }
  else {
    s = car.s;
    s_dot = car.speed;
    d = car.d;
    d_dot = 0;
    d_ddot = 0;
  }
  s_ddot = target.accel;
  
  double t = PARAM_DT;
  double prev_s_dot = s_dot;
  for (size_t i = prev_size; i < PARAM_NB_POINTS; i++) {
    s_dot += s_ddot * PARAM_DT;
    if ((target.accel > 0 && prev_s_dot <= target_velocity_ms && s_dot > target_velocity_ms)
        || (target.accel < 0 && prev_s_dot >= target_velocity_ms && s_dot < target_velocity_ms)) {
      s_dot = target_velocity_ms;
    }
    s_dot = std::max(std::min(s_dot, 0.9*PARAM_MAX_SPEED), 0.0);
    s += s_dot * PARAM_DT;

    prev_s_dot = s_dot;

    new_path_s[i] = PointC2(s, s_dot, s_ddot);
    new_path_d[i] = PointC2(d, d_dot, d_ddot);

    std::vector<double> point_xy = map.getXYspline(s, d);

    next_x_vals.push_back(point_xy[0]);
    next_y_vals.push_back(point_xy[1]);

    t += PARAM_DT;
  }
  traj_jmt.trajectory = TrajectoryXY(next_x_vals, next_y_vals);
  traj_jmt.path_sd = TrajectorySD(new_path_s, new_path_d);

  return traj_jmt;
}

// TODO
TrajectoryXY Trajectory::generate_trajectory(Target target, Map &map, CarData const &car, PreviousPath const &previous_path) {
  TrajectoryXY previous_path_xy = previous_path.xy;
  int prev_size = previous_path.num_xy_reused;
  
  std::vector<double> previous_path_x = previous_path_xy.x_vals;
  std::vector<double> previous_path_y = previous_path_xy.y_vals;

  std::vector<double> ptsx;
  std::vector<double> ptsy;

  double ref_x = car.x;
  double ref_y = car.y;
  double ref_yaw = deg2rad(car.yaw);
  // assign offset of x and y coordinates to each list
  if (prev_size < 2) {
    double prev_car_x = car.x - cos(car.yaw);
    double prev_car_y = car.y - sin(car.yaw);

    ptsx.push_back(prev_car_x);
    ptsx.push_back(car.x);
    ptsy.push_back(prev_car_y);
    ptsy.push_back(car.y);
  }
  else {
    ref_x = previous_path_x[prev_size-1];
    ref_y = previous_path_y[prev_size-1];

    double ref_x_prev = previous_path_x[prev_size-2];
    double ref_y_prev = previous_path_y[prev_size-2];

    ref_yaw = std::atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

    ptsx.push_back(ref_x_prev);
    ptsx.push_back(ref_x);
    ptsy.push_back(ref_y_prev);
    ptsy.push_back(ref_y);
  }
  // stimulate each distance of spline and store these to waypoint list 
  std::vector<double> next_wp0 = map.getXYspline(car.s+30, get_dcenter(target.lane));
  std::vector<double> next_wp1 = map.getXYspline(car.s+60, get_dcenter(target.lane));
  std::vector<double> next_wp2 = map.getXYspline(car.s+90, get_dcenter(target.lane));

  ptsx.push_back(next_wp0[0]);
  ptsx.push_back(next_wp1[0]);
  ptsx.push_back(next_wp2[0]);

  ptsy.push_back(next_wp0[1]);
  ptsy.push_back(next_wp1[1]);
  ptsy.push_back(next_wp2[1]);

  for (size_t i = 0; i < ptsx.size(); i++) {
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;
    ptsx[i] = shift_x * cos(-ref_yaw) - shift_y * sin(-ref_yaw);
    ptsy[i] = shift_x * sin(-ref_yaw) + shift_y * cos(-ref_yaw);
  }
  
  tk::spline spline;
  spline.set_points(ptsx, ptsy);

  std::vector<double> next_x_vals;
  std::vector<double> next_y_vals;

  for (size_t i = 0; i < prev_size; i++) {
    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  double target_x = 30.0;
  double target_y = spline(target_x);
  double target_dist = sqrt(pow(target_x, 2) + pow(target_y, 2));
  
  double x_add_on = 0.0;

  // fill up the rest of the planning points
  // always output 50 points
  for (size_t i = 1; i <= PARAM_NB_POINTS; i++) {
    double N = target_dist / (PARAM_DT * mph_to_ms(target.velocity));
    double x_point = x_add_on + target_x/N;
    double y_point = spline(x_point);

    x_add_on = x_point;

    double x_ref = x_point;
    double y_ref = y_point;

    x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
    y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

    x_point += ref_x;
    y_point += ref_y;

    next_x_vals.push_back(x_point);
    next_y_vals.push_back(y_point);
  }
  return TrajectoryXY(next_x_vals, next_y_vals);
}

