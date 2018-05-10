#include <iostream>
#include <algorithm>
#include <vector>
#include "norm.h"

using namespace std;

/**
 * initialize priors
 */
vector<float> init_priors(unsigned int map_size, vector<float> landmark_positions, float control_stdev);

/**
 * motion model
 */
float motion_model(float pseudo_position, float movement, vector<float> priors, unsigned int map_size, int control_stdev);

int main() {
  float control_stdev = 1.0f;
  // meters vehicle moves per time step 
  float movement_per_timestamp = 1.0f;
  // number of x position on map
  unsigned int map_size = 25;
  // initialize landmark 
  vector<float> landmark_positions {5, 10, 20};
  // initialize priors
  vector<float> priors = init_priors(map_size, landmark_positions, control_stdev);

  for (unsigned int i = 0; i < map_size; i++) {
    float pseudo_position = float(i);
    // get the motion model probability for each x position
    float motion_prob = motion_model(pseudo_position, movement_per_timestamp, priors, map_size, control_stdev);
    cout << "pseudo_position: " << pseudo_position << ", " << "motion_prob: " << motion_prob << endl; 
  }
  return 0;
}

float motion_model(float pseudo_position, float movement, 
        vector<float> priors, unsigned int map_size, int control_stdev) {
    float position_prob = 0.0f;
    for (unsigned int j=0; j < map_size; j++) {
      float next_pesudo_position = float(j);
      float distance_ij = pseudo_position - next_pesudo_position;
      // transition probability
      // The distibution generated consist of standard variance given by control, and mean value given by mean of movement in unit time.
      float transition_prob = normpdf(distance_ij, movement, control_stdev); 
      position_prob += transition_prob * priors[j];
    }
    return position_prob;
}

vector<float> init_priors(unsigned int map_size, vector<float> landmark_positions, float control_stdev) {
  vector<float> priors(map_size, 0.0);
  // normal distribution value of 2 times standard variance
  // plus 1 means due to this distribution's centre is 1, so have to shift 1 unit.
  float normalization_term = landmark_positions.size() * (control_stdev * 2 + 1);
  for (unsigned int i = 0; i < landmark_positions.size(); i++) {
    int landmark_centre = landmark_positions[i];
    priors[landmark_centre] = 1.0f/normalization_term;
    priors[landmark_centre-1] = 1.0f/normalization_term;
    priors[landmark_centre+1] = 1.0f/normalization_term; 
  }
  return priors;
}


