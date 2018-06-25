#ifndef PATH_PLANNING_MAP
#define PATH_PLANNING_MAP
#include "spline.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

class Map {

public:
  Map() {};
  virtual ~Map() {};

  std::vector<double> getFrenet(double x, double y, double theta);
  std::vector<double> getXY(double s, double d);
  std::vecotr<double> getXYspline(double s, double d);
  double getSpeedToFrenet(double v_xy, double s);

  void plot(void);
  double testError(double x, double y, double yaw);
  void read(std::string map_file);

private:
  tk::spline spline_x;
  tk::spline spline_y;
  tk::spline spline_dx;
  tk::spline spline_dy;
  
  std::vector<double> map_waypoints_x;
  std::vector<double> map_waypoints_y;
  std::vector<double> map_waypoints_s;
  std::vector<double> map_waypoints_d;
  std::vector<double> map_waypoints_dx;
  std::vector<double> map_waypoints_dy;
  std::vector<double> map_waypoints_normx;
  std::vector<double> map_waypoints_normy;

  std::vector<double> map_s;

  std::vector<double> map_new_waypoints_x;
  std::vector<double> map_new_waypoints_y;
  std::vector<double> map_new_waypoints_dx;
  std::vector<double> map_new_waypoints_dy;

  std::vector<double> new_map_s;

  double max_error = 0.0;
  double sum_error = 0.0;
  double avg_error = 0.0;
  unsigned int num_error = 0;

  int closest_waypoint(double x, double y, const std::vector<double> &maps_x, const std::vector<double> &maps_y);
  int next_waypoint(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y);

}

#endif // PATH_PLANNING_MAP
