#include "head/utility.h"
#include "head/params.h"
#include "head/map.h"
//#include "head/matplotlibcpp.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <time.h>

//namespace plt = matplotlibcpp;

double MAX_S;

// read the map file
void Map::read(std::string map_file) {
  std::ifstream _in_map(map_file.c_str(), std::ifstream::in);
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
  // tackle exception
  assert(map_waypoints_x.size() && "map file failed to load, please check the path is valid");
  
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

  frenet_s = 0.0;
  new_map_s.push_back(0.0);
  for (size_t i = 1; i < new_map_waypoints_x.size(); i++) {
    frenet_s += distance(new_map_waypoints_x[i], new_map_waypoints_y[i], new_map_waypoints_x[i-1], new_map_waypoints_y[i-1]);
    // 1 point every meter
    new_map_s.push_back(i);
    std::cout << "i: " << i << ", frenet_s: " << frenet_s << std::endl; 
  }
  
  max_error = 0.0;
  sum_error = 0.0;
  avg_error = 0.0;
  num_error = 0;
}

// plot the diagram
/*
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
*/

// find the closest way point
int Map::closest_waypoint(double x, double y, const std::vector<double> &maps_x, const std::vector<double> &maps_y) {
  double closestLen = 100000.0; // just a large number
  int closestWaypoint = 0;
  int size = maps_x.size();

  if (size <= 200) {
    for (size_t i = 0; i < size; i++) {
      double map_x = maps_x[i];
      double map_y = maps_y[i];
      double dist = distance(x, y, map_x, map_y);
      if (dist < closestLen) {
        closestLen = dist;
        closestWaypoint = i;
      }
    }
  }
  else { // faster search
    // 1 jump point with a 181 points map
    int jump_points = size / 181;
    int point = 0;
    while (point < size) {
      double map_x = maps_x[point];
      double map_y = maps_y[point];
      double dist = distance(x, y, map_x, map_y);
      if (dist < closestLen) {
        closestLen = dist;
        closestWaypoint = point;
      }
      point += jump_points;
    }
    // search a point within the nearest refined area
    for (size_t i = closestWaypoint - 91; i < closestWaypoint + 91; i++) {
      int idx = i;
      if (i < 0) {
        idx += size;
      }
      else if (i >= size) {
        idx -= size;
      }

      double map_x = maps_x[idx];
      double map_y = maps_y[idx];
      double dist = distance(x, y, map_x, map_y);
      if (dist < closestLen) {
        closestLen = dist;
        closestWaypoint = idx;
      }
    }
  }
  return closestWaypoint;
}

// get the next point
int Map::next_waypoint(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y) {
  int closestWaypoint = closest_waypoint(x, y, maps_x, maps_y);
  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];
  double heading = std::atan2((map_y - y), (map_x-x));
  double angle = std::fabs(theta - heading);
  angle = std::min(2*M_PI - angle, angle);

  if (angle > M_PI/4) {
    closestWaypoint++;
    if (closestWaypoint == maps_x.size()) {
      closestWaypoint = 0;
    }
  }
  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
std::vector<double> Map::getFrenet(double x, double y, double theta) {
  std::vector<double> &maps_s = this->new_map_s;
  std::vector<double> &maps_x = this->new_map_waypoints_x;
  std::vector<double> &maps_y = this->new_map_waypoints_y;

  int next_wp = next_waypoint(x, y, theta, maps_x, maps_y);
  int prev_wp = (next_wp != 0) ? next_wp - 1 : maps_x.size()-1;

  double n_x = maps_x[next_wp] - maps_x[prev_wp];
  double n_y = maps_y[next_wp] - maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // projection of x onto n
  double proj_norm = (n_x*x_x + n_y*x_y) / (std::pow(n_x, 2) + std::pow(n_y, 2));
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x, x_y, proj_x, proj_y);

  if (PARAM_MAP_BOSCH == false) {
    double center_x = PARAM_CENTER_X - maps_x[prev_wp];
    double center_y = PARAM_CENTER_Y - maps_y[prev_wp];
    double centerToPos = distance(center_x, center_y, x_x, x_y);
    double centerToRef = distance(center_x, center_y, proj_x, proj_y);

    if (centerToPos <= centerToRef) {
      frenet_d *= -1;
    }
  }

  double frenet_s = maps_s[prev_wp];
  frenet_s += distance(0, 0, proj_x, proj_y);
  assert(frenet_d >= 0);
  
  return {frenet_s, frenet_d};
}


// Transform Frenet coordinates to Cartesian space
std::vector<double> Map::getXY(double s, double d) {
  std::vector<double> &maps_s = map_waypoints_s;
  std::vector<double> &maps_x = map_waypoints_x;
  std::vector<double> &maps_y = map_waypoints_y;
  int prev_wp = -1;
  while (s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1))) {
    prev_wp++;
  }
  int wp2 = (prev_wp+1)%maps_x.size();
  double heading = std::atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
  double seg_s = s - maps_s[prev_wp];
  double seg_x = maps_x[prev_wp] + seg_s*std::cos(heading);
  double seg_y = maps_y[prev_wp] + seg_s*std::sin(heading);
  double perp_heading = heading - M_PI/2;
  double x = seg_x + d*std::cos(perp_heading);
  double y = seg_y + d*std::sin(perp_heading);

  return {x, y};
}

// get the spline coordinate
std::vector<double> Map::getXYspline(double s, double d) {
  s = std::fmod(s, MAX_S);
  double x = spline_x(s) + d * spline_dx(s);
  double y = spline_y(s) + d * spline_dy(s);
  return {x, y};
}

// transform the speed vector to Frenet space
double Map::getSpeedToFrenet(double v_xy, double s) {
  s = std::fmod(s, MAX_S);
  double dx_over_ds = spline_x.deriv(1, s);
  double dy_over_ds = spline_y.deriv(1, s);
  double vs = v_xy / std::sqrt(std::pow(dx_over_ds, 2) + std::pow(dy_over_ds, 2));
  return vs;
}

// Calculating the distance from the realistic to frenet coordinates. 
double Map::testError(double car_x, double car_y, double car_yaw) {
  double error = 0.0;
  clock_t start = clock();
  std::vector<double> frenet = getFrenet(car_x, car_y, deg2rad(car_yaw));
  
  int frenet_s = frenet[0];
  int frenet_d = frenet[1];

  std::vector<double> car_xy = getXYspline(frenet_s, frenet_d);
  clock_t stop = clock();
  double elapsed = (double)(stop-start)*1000000.0 / CLOCKS_PER_SEC;

  error = distance(car_xy[0], car_xy[1], car_x, car_y);
  sum_error += error;
  num_error++;
  avg_error = sum_error/num_error;

  if (error > max_error) {
    max_error = error;
  }

  std::cout << "error=" << error << ", elapsed=" << elapsed << "us, (max_error=" << max_error << ", avg_error=" << avg_error << ")" << std::endl;
  
  return error;
}
