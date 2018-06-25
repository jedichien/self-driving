#ifndef PATH_PLANNING_PREDICTION
#define PATH_PLANNING_PREDICTION

#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include "utility.h"
#include "params.h"

class Prediction {
  public:
    Prediction(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car, int horizon);
    virtual ~Prediction();

    std::map<int, std::vector<Coord>> get_predictions() const {
      return _predictions;
    }
    
    double get_safety_distance() const {
      return _safety_distance;
    }

    double get_paranoid_safety_distance() const {
      return _paranoid_safety_distance;
    }

    double get_lane_speed(int lane) const;
  
    double get_lane_free_space(int lane) const;

 
  private:
    void set_safety_distances(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car);

    void set_lane_info(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car);

    std::vector<int> find_closest_objects(std::vector<std::vector<double>> const &sensor_fusion, CarData const &car);

    double get_safety_distance(double vel_back, double vel_front, double time_latency);

    // INITIALIZE
    std::vector<int> _front = {-1, -1, -1};
    std::vector<int> _back = {-1, -1, -1};

    std::vector<double> _front_dmin = {INF, INF, INF};
    std::vector<double> _back_dmin = {INF, INF, INF};

    std::vector<double> _front_velocity = {PARAM_MAX_SPEED, PARAM_MAX_SPEED, PARAM_MAX_SPEED};
    std::vector<double> _front_safety_distance = {PARAM_SD_LC, PARAM_SD_LC, PARAM_SD_LC};

    std::vector<double> _back_velocity = {PARAM_MAX_SPEED, PARAM_MAX_SPEED, PARAM_MAX_SPEED};
    std::vector<double> _back_safety_distance = {PARAM_SD_LC, PARAM_SD_LC, PARAM_SD_LC};

    // at most 6 predictions of horizontal coordinates
    std::map<int, std::vector<Coord>> _predictions;    
    
    double _lane_speed[PARAM_NB_LANES];
    double _lane_free_space[PARAM_NB_LANES];
    
    // safety distance
    double _vel_ego;
    double _decel;

    // front distance
    double _dist_front;
    double _vel_front;

    // latency
    double _time_to_collision;
    double _time_to_stop;
    double _time_to_decelerate;
   
    // used by behavior planner
    double _safety_distance = PARAM_SD;
    double _paranoid_safety_distance = PARAM_SD;
};

#endif // PATH_PLANNING_PREDICTION
