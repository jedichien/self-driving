#include <iostream>
#include <algorithm>
#include <vector>
#include "normpdf.h"

using namespace std;

vector<float> pseudo_range_estimator(vector<float> landmark_positions, float pseudo_position, float distance_max);

float observation_model(vector<float> landmark_positions, vector<float> observations, 
        vector<float> pseudo_ranges, float observation_stdev);

int main() {
  // standard deviation
  float observation_stdev = 1.0f;
  unsigned int map_size = 25;
  float distance_max = map_size;
  vector<float> landmark_positions {5.f, 10.f, 12.f, 20.f};
  // practicle observations range
  vector<float> observations {5.5f, 13.f, 15.f};

  for (unsigned int i = 0; i < map_size; ++i) {
    float pseudo_position = float(i);
    vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, pseudo_position, distance_max);
    // get observation probability
    float observation_prob = observation_model(landmark_positions, observations, pseudo_ranges, observation_stdev);
    cout << "pseudo_position: " << pseudo_position << ", observation_prob: " << observation_prob << endl; 
  }
  return 0;
}

float observation_model(vector<float> landmark_positions, vector<float> observations,
        vector<float> pseudo_ranges, float observation_stdev) {
  float distance_prob = 1.0f;
  for (unsigned int z = 0; z < observations.size(); ++z) {
    float pseudo_range_min = pseudo_ranges[z];
    // assume the noise behavior of the individual range value from 0~N is independent
    // porbability based on practicle range of observation distribution
    distance_prob *= normpdf(observations[z], pseudo_range_min, observation_stdev);
  }
  return distance_prob;
}

vector<float> pseudo_range_estimator(vector<float> landmark_positions, float pseudo_position, float distance_max) {
  vector<float> pseudo_ranges;
  for (unsigned int l = 0; l < landmark_positions.size(); ++l) {
    // `estimate` pseudo range for each single landmark
    float range_l = landmark_positions[l] - pseudo_position;
    if (range_l > 0.0f) {
      pseudo_ranges.push_back(range_l);
    }
    else {
      pseudo_ranges.push_back(distance_max);
    }
  }
  sort(pseudo_ranges.begin(), pseudo_ranges.end()); 
  return pseudo_ranges;
}

