#include "head/behavior.h"

// contructor
Behavior::Behavior(std::vector<std::vector<double>> const &sensor_fusion, CarData car, Prediction const &prediction) {
  Target target;
  target.time = 2.0;
  
  double car_speed_target = (car.emergency) ? car.speed : car.speed_target;
  double safety_distance = prediction.get_safety_distance();
  bool too_close = false;
  int ref_vel_inc = 0; // 0 for constant speed, -1 for max deceleration, 1 for max acceleration
  double ref_vel_ms = mph_to_ms(car_speed_target);
  double closest_speed_ms = PARAM_MAX_SPEED;
  double closest_dist = INF;

  // Phase: 1
  // get the closest distance within sensor fusion datas.
  for (size_t i = 0; i < sensor_fusion.size(); i++) {
    float d = sensor_fusion[i][6];
    // approaching to our car
    if (d > get_dlef(car.lane) && d < get_dright(car.lane)) {
      // initial
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = std::sqrt(std::pow(vx, 2) + std::pow(vy, 2));
      double check_car_s = sensor_fusion[i][5];
      std::cout << "obj_idx=" << i << " REF_VEL_MS=" << ref_vel_ms << " CHECK_SPEED=" << check_speed << std::endl;
      // action if our car beyond safety gaps to others.
      if ((check_car_s > car.s) && ((check_car_s - car.s) < safety_distance)) {
        too_close = true;
        double dist_to_check_car_s = check_car_s - car.s;
        if (dist_to_check_car_s < closest_dist) {
          closest_dist = dist_to_check_car_s;
          closest_speed_ms = check_speed;
        }
      }
    }
  }

  // Phase: 2
  // I call this, "slow down strategy".  
  // if we nearly approach to others, we should slow down and decelerate.
  if (too_close) {
    if (ref_vel_ms > closest_speed_ms) {
      // slow down
      car_speed_target -= PARAM_MAX_SPEED_INC_MPH;
      if (closest_dist <= 10 && car_speed_target > closest_speed_ms) {
        car_speed_target -= 5 * PARAM_MAX_SPEED_INC_MPH;
      }
    }
    car_speed_target = std::max(car_speed_target, 0.0);
    ref_vel_inc = -1;
  } // still under control
  else if (car_speed_target < PARAM_MAX_SPEED_MPH) {
    // throttle a little bit
    car_speed_target += PARAM_MAX_SPEED_INC_MPH;
    car_speed_target = std::min(car_speed_target, PARAM_MAX_SPEED_MPH);
    ref_vel_inc = 1;
  }

  // Phase: 3
  // tackle too close problem
  target.lane = car.lane;
  target.velocity = car_speed_target;
  if (fabs(car.d - get_dcenter(car.lane)) <= 0.01) {
    target.time = 0.0;
    target.velocity = ms_to_mph(closest_speed_ms);
    target.accel = 0.7 * PARAM_MAX_ACCEL;
    double car_speed_ms = mph_to_ms(car.speed);
    if (closest_speed_ms < car_speed_ms && closest_dist <= safety_distance) {
      target.accel *= -1;
    }
  }
  // add this to candidate
  _targets.push_back(target);

  // Phase: 4
  // provide trajectory with feasible choices
  // compose of lane, speed 
  target.velocity = car_speed_target;
  target.time = 2.0;
  
  std::vector<int> backup_lanes;
  if (car.lane < 2) {
    backup_lanes.push_back(car.lane + 1);
  }
  if (car.lane > 0) {
    backup_lanes.push_back(car.lane - 1);
  }
  
  std::vector<double> backup_vel;
  switch (ref_vel_inc) {
    case 1:
      backup_vel.push_back(car_speed_target - PARAM_MAX_SPEED_INC_MPH);
      backup_vel.push_back(car_speed_target - 2 * PARAM_MAX_SPEED_INC_MPH);
      break;
    case 0:
      backup_vel.push_back(car_speed_target - PARAM_MAX_SPEED_INC_MPH);
      break;
    case -1:
      backup_vel.push_back(car_speed_target - PARAM_MAX_SPEED_INC_MPH);
      break;
  }
  
  // backup some stuff
  // TODO: I still figure it out the reason why we should need this.
  target.lane = car.lane;
  for (auto bk_velocity : backup_vel) {
    target.velocity = bk_velocity;
    _targets.push_back(target);
  }
  
  // backup velocities on backup lanes
  for (auto bk_velocity : backup_vel) {
    target.velocity = bk_velocity;
    for (auto bk_lane : backup_lanes) {
      target.lane = bk_lane;
      _targets.push_back(target);
    }
  }
  
  // last target candidate while emergency situation occurs
  target.lane = car.lane;
  target.velocity = predictions.get_lane_speed(car.lane);
  target.time = 0.0;
  target.accelerate = -0.85 * PARAM_MAX_ACCEL;
  _targets.push_back(target);
}

// destructor
Behavior::~Behavior() {}
