#ifndef PATH_PLANNING_PARAMS
#define PATH_PLANNING_PARAMS

#include <string>
#include "utility.h"

const bool PARAM_MAP_BOSCH = false;

const std::string _map_file = "../data/highway_map.csv";
const std::string _map_bosch_file = "../data/highway_map_bosch1_final.csv";

const double MAXIMUM_S = 6945.554;
extern double MAX_S;

// Coordinates
const double PARAM_CENTER_X = 1000;
const double PARAM_CENTER_Y = 2000;
const int PARAM_NB_POINTS = 50; // considerable number
const double PARAM_DT     = 0.02; // 1 point per 0.02s
const double PARAM_LANE_WIDTH  = 4.0; // meters

// Speed
const double PARAM_MAX_SPEED_MPH = 49;
const double PARAM_MAX_SPEED     = 22;
const double PARAM_MAX_ACCEL     = 10;
const double PARAM_MAX_JERK      = 10;
const double PARAM_FOV           = 70.0; // Field of View
const double PARAM_MAX_SPEED_INC     = PARAM_MAX_ACCEL * PARAM_DT;
const double PARAM_MAX_SPEED_INC_MPH = ms_to_mph(PARAM_MAX_SPEED_INC);

// Safety parameters
const double PARAM_DIST_SAFETY      = 3.5;
const int PARAM_PREV_PATH_XY_REUSED = 5;
const bool PARAM_TRAJECTORY_JMT     = true;
const double PARAM_CAR_SAFETY_W     = 3;
const double PARAM_CAR_SAFETY_L     = 6;
const int PARAM_MAX_COLLISION_STEP  = 25;

// cost.cpp: weighted of cost function
const int PARAM_COST_FEASIBILITY = 10000;
const int PARAM_COST_SAFETY      = 1000;
const int PARAM_COST_LEGALITY    = 100;
const int PARAM_COST_COMFORT     = 10;
const int PARAM_COST_EFFICIENCY  = 1;

// Safety Distance
const int PARAM_NB_LANES = 3;
// default safety distance for "LANE CHANGING"
const double PARAM_SD_LC = 10.0;
// defualt safety distance
const double PARAM_SD    = 30.0;

#endif // PATH_PLANNING_PARAMS
