#include "head/prediction.h"
// constructor
Prediction::Prediction(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car, int horizon) {
  std::map<int, std::vector<Coord>> prediction;
  std::vector<int> closest_objects = find_closest_objects(sensor_fusion, car);
  // extract fusion
  for (auto &fusion_index : closest_objects) {
    if (fusion_index >= 0) {
      double x = sensor_fusion[fusion_index][1];
      double y = sensor_fusion[fusion_index][2];
      double vx = sensor_fusion[fusion_index][3];
      double vy = sensor_fusion[fusion_index][4];
      std::vector<Coord> prediction;
      for (size_t j = 0; j < horizon; j++) {
        Coord coord;
        coord.x = x + vx * j * PARAM_DT;
        coord.y = y + vy * j * PARAM_DT;
        prediction.push_back(coord);
      }
      _predictions[fusion_index] = prediction;
    }
  }
  set_safety_distances(sensor_fusion, car);
  set_lane_info(sensor_fusion, car);
}

// destructor
Prediction::~Prediction() {}

double get_sensor_fusion_vel(std::vector<std::vector<double>> const &sensor_fusion, int idx, double default_vel) {
  double vx, vy, vel;
  if (idx >= 0 && idx < sensor_fusion.size()) {
    vx = sensor_fusion[idx][3];
    vy = sensor_fusion[idx][4];
    vel = sqrt(pow(vx, 2) + pow(vy, 2));
  }
  else {
    vel = default_vel;
  }
  return vel;
}

/**
 * Get Safety Distance
 * get safety distance according to front vehicle
 *
 * @param vel_back back of vehicle speed
 * @param vel_front front of vehicle speed
 * @param time_latency latency time
 */
double Prediction::get_safety_distance(double vel_back, double vel_front, double time_latency) {
  double safety_distance = PARAM_SD_LC;
  if (vel_back > vel_front) {
    double time_to_decelerate = (vel_back - vel_front) / _decel + time_latency;
    safety_distance = std::max(vel_back * time_to_decelerate + 1.5 * PARAM_CAR_SAFETY_L, PARAM_SD_LC);
  }
  return safety_distance;
}

/**
 * Set Safety Distance
 * apply safety distance in our car
 *
 * @param sensor_fusion sensor fusion
 * @param car information of vehicle
 */
void Prediction::set_safety_distances(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car) {
  _vel_ego = mph_to_ms(car.speed);
  _decel = 0.8 * PARAM_MAX_ACCEL;
  _time_to_stop = _vel_ego / _decel;

  _vel_front = get_sensor_fusion_vel(sensor_fusion, _front[car.lane], PARAM_MAX_SPEED);
  _dist_front = _front_dmin[car.lane];
  
  // estimate safety distance
  if (_vel_ego > _vel_front) {
    _time_to_collision = _dist_front / (_vel_ego - _vel_front);
    _time_to_decelerate = (_vel_ego - _vel_front) / _decel;
    _safety_distance = _vel_ego *  _time_to_decelerate + 1.75 * PARAM_CAR_SAFETY_L;
  }
  else { // current speed is still beyond the front
    _time_to_collision = INF;
    _time_to_decelerate = 0;
    _safety_distance = 1.75 * PARAM_CAR_SAFETY_L;
  }

  _paranoid_safety_distance = _vel_ego * _time_to_stop + 2 * PARAM_CAR_SAFETY_L;

  //std::cout << "SAFETY: D=" << _distance_front << " dV=" << _vel_ego - _vel_front << " TTC=" << _time_to_collision << " TTD=" << _time_to_decelerate << " SD=" << _safety_distance << " PSD=" << _paranoid_safety_distance << std::endl;

  for (size_t i = 0; i < PARAM_NB_LANES; i++) {
    // front safety distance
    _front_velocity[i] = get_sensor_fusion_vel(sensor_fusion, _front[i], PARAM_MAX_SPEED);
    _front_safety_distance[i] = get_safety_distance(_vel_ego, _front_velocity[i], 0.0);
    // back safety distance
    _back_velocity[i] = get_sensor_fusion_vel(sensor_fusion, _back[i], 0);
    _back_safety_distance[i] = get_safety_distance(_back_velocity[i], _vel_ego, 2.0);

    //std::cout << "SAFETY_DISTANCE for LC[" << i << "]: front_sd=" << _front_safety_distance[i] << " back_sd=" << back_safety_distance << std::endl;
  }
}

/**
 * Set the lane information, including front and back vehicle's
 * regulate the distance an speed if needed
 * 
 * @param sensor_fusion sensor fusion
 * @param car information of vehicle
 */
void Prediction::set_lane_info(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car) {
  int car_lane = get_lane(car.d);

  for (size_t i = 0; i < _front.size(); i++) {
    std::cout << "lane " << i << ": ";
    std::cout << "front " << _front[i] << " at " << _front_dmin[i] << " s_meters; ";
    std::cout << "back " << _back[i] << " at " << _back_dmin[i] << " s_meters" << std::endl;
    int lane = i;
    // vehicle in front of us
    if (_front[i] >= 0) {
      // if it too close
      if (lane != car_lane && (_back_dmin[i] <= _back_safety_distance[i] || _front_dmin[i] <= _front_safety_distance[i])) {
        _lane_speed[i] = 0;
        _lane_free_space[i] = 0;
      }
      else {
        double vx = sensor_fusion[_front[i]][3];
        double vy = sensor_fusion[_front[i]][4];
        _lane_speed[i] = sqrt(pow(vx, 2) + pow(vy, 2));
        _lane_free_space[i] = _front_dmin[i];
      }
    }
    else {
      if (lane != car_lane && _back_dmin[i] <= _back_safety_distance[i]) {
        _lane_speed[i] = 0;
        _lane_free_space[i] = 0;
      }
      else {
        _lane_speed[i] = PARAM_MAX_SPEED_MPH;
        _lane_free_space[i] = PARAM_FOV;
      }
    }
    std::cout << "Prediction::_lane_speed[" << i << "]=" << _lane_speed[i] << std::endl;
  }
}

/**
 * Updating the front and back information, retrieve the closest ones.
 *
 * @param sensor_fusion
 * @param car vehicle information
 */
std::vector<int> Prediction::find_closest_objects(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car) {
  double sfov_min = car.s - PARAM_FOV;
  double sfov_max = car.s + PARAM_FOV;
  double sfov_shift = 0;
  
  if (sfov_min < 0) {
    sfov_shift = -sfov_min;
  }
  else if (sfov_max > MAX_S) {
    sfov_shift = MAX_S - sfov_max;
  }

  sfov_min += sfov_shift;
  sfov_max += sfov_shift;
  // prevent illegal conditions
  assert(sfov_min >= 0 && sfov_min <= MAX_S);
  assert(sfov_max >= 0 && sfov_max <= MAX_S);

  double car_s = car.s + sfov_shift;
  
  for (size_t i = 0; i < sensor_fusion.size(); i++) {
    double s = sensor_fusion[i][5] + sfov_shift;
    // vehicle in FOV
    if (s >= sfov_min && s <= sfov_max) {
      double d = sensor_fusion[i][6];
      int lane = get_lane(d);
      if (lane < 0 || lane > 2) continue;
      
      double dist = fabs(s - car_s);
      // update minimum distance to the others
      // front
      if (s >= car_s) {
        if (dist < _front_dmin[lane]) {
          _front[lane] = i;
          _front_dmin[lane] = dist;
        }
      }
      else {
        if (dist < _back_dmin[lane]) {
          _back[lane] = i;
          _back_dmin[lane] = dist;
        }  
      }
    }    
  }
  return {
    _front[0], _back[0],
    _front[1], _back[1],
    _front[2], _back[2]
  };
}

/**
 * Measuring the spedd of lane.
 * @param lane lane id
 */
double Prediction::get_lane_speed(int lane) const {
  if (lane >= 0 && lane <= 3) {
    return _lane_speed[lane];
  }
  return 0;
}

/**
 * Measuring the free space from ego to lane.
 * @param lane lane id
 */
double Prediction::get_lane_free_space(int lane) const {
  if (lane >= 0 && lane <= 3) {
    return _lane_free_space[lane];
  }
  return 0;
}

