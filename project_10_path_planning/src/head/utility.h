#ifndef PATH_PLANNING_UTILITY
#define PATH_PLANNING_UTILITY

#include <vector>
#define INF 1e10

// ========================
// Coordinates and Distance
// ========================
typedef std::vector<double> t_coord;
typedef std::vector<t_coord> t_traj;
typedef std::vector<t_traj> t_trajSet;

double deg2rad(double x);
double rad2deg(double x);
double mph_to_ms(double mph);
double ms_to_mph(double ms);

// coordinates of "left" lane. 
double get_dlef(int lane);
// coordinates of "right" lane
double get_dright(int lane);
// coordinates of "center" lane
double get_dcenter(int lane);

int get_lane(double d);
double distance(double x1, double y1, double x2, double y2);

// ====================
// Vehicle Data
// ====================
enum {
  ID = 0,
  X  = 1,
  Y  = 2,
  VX = 3,
  VY = 4,
  S  = 5,
  D  = 6,
  SIZE = 7
};

struct Coord {
  double x;
  double y;
};

struct Frenet {
  double s;
  double d;
};

struct CarData {
  double x;
  double y;
  double s;
  double d;
  double yaw;
  double speed;
  double speed_target;
  int lane;
  bool emergency;
  
  CarData(double X=0, double Y=0, double S=0, 
          double D=0, double YAW=0, double V=0, 
          double VF=0, double L=0, bool E=false) : x(X), y(Y), s(S), yaw(YAW), speed(V), speed_target(VF), lane(L), emergency(E) {}

};

#endif // PATH_PLANNING_UTILITY
