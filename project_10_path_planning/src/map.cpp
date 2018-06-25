#include "head/utility.h"
#include "head/params.h"
#include "head/map.h"
#include "head/matplotlibcpp.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <time.h>

namespace plt = matplotlibcpp;

double MAX_S;

void Map::read(std::string map_file) {
  ifstream _in_map(map_file.c_str(), ifstream::in);
  std::string line;
  bool not_started = true;
  double x0, y0, dx0, dy0;
  double last_s = 0.0;
  
  while (std::getline(_in_map, line)) {
    std::istringstream iss(line);
    double x, y;
    float s, d_x, d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    if (not_started) {
      x0 = x;
      y0 = y;
      dx0 = d_x;
      dy0 = d_y;
      not_started = false;
    }

    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    last_s = s;
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
    map_waypoints_normx.push_back(x+10*d_x);
    map_waypoints_normy.push_back(y+10*d_y);
  }
  // tackle exeptional situation
  assert(map_waypoints.size() && "map file failed to load, please check the path is valid");
  
  MAX_S = (PARAM_MAP_BOSCH == true) ? last_s : MAXIMUM_S;
  // only for spline
  if (PARAM_MAP_BOSCH == false) {
    map_waypoints_x.push_back(x0);
    map_waypoints_y.push_back(y0);
    map_waypoints_s.push_back(MAX_S);
    map_waypoints_dx.push_back(dx0);
    map_waypoints_dy.push_back(dy0);
    map_waypoints_normx.push_back(x0+10*dx0);
    map_waypoints_normy.push_back(y0+10*dy0);
  }
  
  spline_x.set_points(map_waypoints_s, map_waypoints_x);
  spline_y.set_points(map_waypoints_s, map_waypoints_y);
  spline_dx.set_points(map_waypoints_s, map_waypoints_dx);
  spline_dy.set_points(map_waypoints_s, map_waypoints_dy);
   
  if (PARAM_MAP_BOSCH == false) {
    map_waypoints_x.pop_back();
    map_waypoints_y.pop_back();
    map_waypoints_s.pop_back();
    map_waypoints_dx.pop_back();
    map_waypoints_dy.pop_back();
    map_waypoints_normx.pop_back();
    map_waypoints_normy.pop_back();
  }

  double len_ref = 0.0;
  double prev_x = spline_x(0);
  double prev_y = spline_y(0);
  for (double s = 1; s <= floor(MAX_S); s++) {
    double x = spline_x(s);
    double y = spline_y(s);
    double dx = spline_dx(s);
    double dy = spline_dy(s);
    
    new_map_waypoints_x.push_back(x);
    new_map_waypoints_y.push_back(y);
    new_map_waypoints_dx.push_back(dx);
    new_map_waypoints_dy.push_back(dy);
  }

  double frenet_s = 0.0;
  map_s.push_back(0.0);
  for (size_t i = 1; i < map_waypoints_x.size(); i++) {
    frenet_s += distance(map_waypoints_x[i], map_waypoints_y[i], map_waypoints_x[i-1], map_waypoints_y[i-1]);
    map_s.push_back(frenet_s);
  }

  frenet = 0.0;
  new_map_s.push_back(0.0);
  for (size_t i = 1; i < new_map_waypoints_x.size(); i++) {
    frenet_s += distance(new_map_waypoints_x[i], new_map_waypoints_y[i], new_map_waypoints_x[i-1], new_map_waypoints_y[i-1]);
    // 1 point every meter
    new_map_s.push_back(i);
    std::cout << "i: " << i << << ", frenet_s: " << frenet_s << std::endl; 
  }
  
  max_error = 0.0;
  sum_error = 0.0;
  avg_error = 0.0;
  num_error = 0;
}

Map::~Map() {}

void Map::plot(void) {
  plt::title("Map");
  plt::plot(map_waypoints_x, map_waypoints_y, "r*");
  plt::plot(map_waypoints_normx, map_waypoints_normy, "g*");
  plt::plot(new_map_waypoints_x, new_map_waypoints_y, "b-");
  std::vector<double> car_x = {1, 770.0906};
  std::vector<double> car_y = {1, 1129.872};
  plt::plot(car_x, car_y, "gx");
  plt::show();
}

// TODO
