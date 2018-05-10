#include <iostream>
#include <algorithm>
#include <vector>
#include "norm.h"

using namespace std;
/**
 * =============
 * motion model
 * =============
 */
vector<float> init_priors(unsigned int map_size, vector<float> landmark_positions, float control_stdev);

float motion_model(float pseudo_position, float movement, vector<float> priors, 
        unsigned int map_size, int control_stdev);

vector<float> pseudo_range_estimator(vector<float> landmark_positions, float pseudo_position, unsigned int map_size);

/**
 * =============
 * observation model
 * =============
 */
vector<float> pseudo_range_estimator(vector<float> landmark_positions, float pseudo_position, unsigned int map_size);

float observation_model(vector<float> landmark_positions, vector<float> observations, vector<float> pseudo_ranges, float observation_stdev);

int main() {
  float control_stdev = 1.0f;
  float movement_per_timestamp = 1.0f;
  float observation_stdev = 1.0f;
  unsigned int map_size = 25;
  float distance_max = map_size;
  vector<float> landmark_positions {3, 9, 14, 23};
  vector<vector<float>> sensor_obs {{1,7,12,21}, {0,6,11,20}, {5,10,19}, {4,9,18},
      {3,8,17}, {2,7,16}, {1,6,15}, {0,5,14}, {4,13},
      {3,12},{2,11},{1,10},{0,9},{8},{7},{6},{5},{4},{3},{2},{1},{0}, {}, {}, {}};
  vector<float> priors = init_priors(map_size, landmark_positions, control_stdev);
  vector<float> posteriors(map_size, 0.0);
  unsigned int time_steps = sensor_obs.size();
  vector<float> observations;
  for (unsigned int t = 0; t < time_steps; t++) {
    if (!sensor_obs[t].empty()) {
      observations = sensor_obs[t];
    }
    else {
      observations = {float(distance_max)};
    }
    for (unsigned int i = 0; i < map_size; i++) {
      float pseudo_position = float(i);
      float motion_prob = motion_model(pseudo_position, movement_per_timestamp, priors, map_size, control_stdev);
      vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, pseudo_position, map_size);
      float observation_prob = observation_model(landmark_positions, observations, pseudo_ranges, observation_stdev);
      posteriors[i] = motion_prob * observation_prob;
    }
    // TODO: normalize vector
    posteriors = normvector(posteriors);
    priors = posteriors;
  }

  for (unsigned int p = 0; p < posteriors.size(); p++) {
    cout << "position: " << p << ", posteriors_prob: " << posteriors[p] << endl;
  }
  return 0;
}

// initialize priors assuming vehicle at landmark +/- 1.0 meters position stdev
vector<float> init_priors(unsigned int map_size, vector<float> landmark_positions, float control_stdev) {
  vector<float> priors(map_size, 0.0);
  // TODO: figure out why?
  float normalization_term = landmark_positions.size() * (control_stdev * 2 + 1);
  for (unsigned int i = 0; i < landmark_positions.size(); i++) {
    int landmark_centre = landmark_positions[i];
    priors[landmark_centre] = 1.0f/normalization_term;
    priors[landmark_centre-1] = 1.0f/normalization_term;
    priors[landmark_centre+1] = 1.0f/normalization_term;
  }
  return priors;
}

float motion_model(float pseudo_position, float movement, vector<float> priors,
        unsigned int map_size, int control_stdev) {
  float position_prob = 0.0f;
  for (unsigned int i = 0; i < map_size; ++i) {
    float next_pseudo_position = float(i);
    float distance_ij = next_pseudo_position - pseudo_position;
    float transition_prob = normpdf(distance_ij, movement, control_stdev);
    position_prob += transition_prob * priors[i];
  }
  return position_prob;
}

vector<float> pseudo_range_estimator(vector<float> landmark_positions, float pseudo_position, unsigned int map_size) {
  vector<float> pseudo_ranges;
  for (unsigned int i = 0; i < landmark_positions.size(); ++i) {
    float range_l = landmark_positions[i] - pseudo_position;
    if (range_l > 0.0f) {
      pseudo_ranges.push_back(range_l);
    }
    else {
      pseudo_ranges.push_back(map_size); 
    }
  }
  sort(pseudo_ranges.begin(), pseudo_ranges.end());
  return pseudo_ranges;
}

float observation_model(vector<float> landmark_positions, vector<float> observations,vector<float> pseudo_ranges, float observation_stdev) {
  float distance_prob = 1.0f;
  for (unsigned int i = 0; i < observations.size(); ++i) {
    float pseudo_ranges_min = pseudo_ranges[i];
    distance_prob *= normpdf(observations[i], pseudo_ranges_min, observation_stdev);
  }
  return distance_prob;

}
