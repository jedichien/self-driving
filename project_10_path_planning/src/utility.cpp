#include "head/utility.h"
#include "head/params.h"
#include <cmath>

#define FACTOR_MS_TO_MPH 2.24 // check this `https://www.quora.com/How-do-I-convert-m-s-to-mph`

// ================
// Orientation
// ================

double deg2rad(double x) {
  return x * M_PI / 180;
}

double rad2deg(double x) {
  return x * 180 / M_PI;
}

// ===================
// Speed unit
// ===================

double mph_to_ms(double mph) {
  return mph / FACTOR_MS_TO_MPH;
}

double ms_to_mph(double ms) {
  return ms * FACTOR_MS_TO_MPH;
}

// ===========================
// Distance
// ===========================

double get_dleft(int lane) {
  double dleft = lane * PARAM_LANE_WIDTH;
  return dleft;
}

double get_dright(int lane) {
  double dright = (lane+1) * PARAM_LANE_WIDTH;
  return dright;
}

double get_dcenter(int lane) {
  double dcenter = (lane + 0.5) * PARAM_LANE_WIDTH;
  // regulate it
  if (dcenter >= 10) {
    dcenter = 9.8;
  }
  return dcenter;
}

int get_lane(double d) {
  return (int)(d / PARAM_LANE_WIDTH);
}

double distance(double x1, double y1, double x2, double y2) {
  return sqrt(pow(x2-x1, 2) + pow(y2-y1, 2));
}

