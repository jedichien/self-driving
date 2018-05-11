/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define EPS 1e-5

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  if (is_initialized) {
    return;
  }
  num_particles = 100;
  // standard deviation
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  // normal distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  // generate particles with normal distribution with mean on GPS values
  for (unsigned int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(unsigned int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;

    if (fabs(yaw_rate) < EPS) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for(auto& observation : observations) {
    double min_dist = numeric_limits<double>::max();
    for(const auto& prediction : predicted) {
      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (distance < min_dist) {
        observation.id = prediction.id;
        min_dist = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    //   Learning resource: 
    //     homogenous tutorial -> https://www.youtube.com/watch?v=tAu8-gkxAcE
    for(unsigned int i = 0; i < num_particles; i++) {
      double particle_x = particles[i].x;
      double particle_y = particles[i].y;
      double particle_theta = particles[i].theta;
      
      vector<LandmarkObs> predictions;

      for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        float lm_x = map_landmarks.landmark_list[j].x_f;
        float lm_y = map_landmarks.landmark_list[j].y_f;
        int lm_id = map_landmarks.landmark_list[j].id_i;
        // only consider landmarks within sensor range of the particle
        if(fabs(lm_x - particle_x) <= sensor_range && fabs(lm_y - particle_y) <= sensor_range) {
          predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
        }
      }
      vector<LandmarkObs> transform_observations;
      for(unsigned int j = 0; j < observations.size(); j++) {
        double t_x = particle_x + cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y;
        double t_y = particle_y + sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y;
        transform_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
      }
      // data association for prediction and observation on current particle
      dataAssociation(predictions, transform_observations);
      particles[i].weight = 1.0;
      for(unsigned int j = 0; j < transform_observations.size(); j++) {
        double o_x, o_y, p_x, p_y;
        o_x = transform_observations[j].x;
        o_y = transform_observations[j].y;
        int asso_predict_id = transform_observations[j].id;
        // (x, y) coordinates of the prediction assocaited with the current observation
        for(unsigned int k = 0; k < predictions.size(); k++) {
          if(predictions[k].id == asso_predict_id) {
            p_x = predictions[k].x;
            p_y = predictions[k].y;
            break;
          }
        }
        // weight for this observation with multivariate Gaussian
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];
          
        double mep_x = pow(p_x-o_x, 2) / (2*pow(std_x, 2));
        double mep_y = pow(p_y-o_y, 2) / (2*pow(std_y, 2));

        double obs_w = (1.0/(2*M_PI*std_x*std_y)) * exp(-(mep_x + mep_y));
        particles[i].weight *= obs_w;
      }
    } // end particles
    double norm_factor = 0.0;
    for(const auto& particle : particles) {
      norm_factor += particle.weight;
    }
    for(auto& particle : particles) {
      particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for(unsigned int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }
  uniform_real_distribution<double> un_double_dist(0.0, max_weight);
  uniform_int_distribution<int> un_int_dist(0, num_particles-1);
  int index = un_int_dist(gen);
  double beta = 0.0;
  vector<Particle> resampledParticles;
  for(unsigned int i = 0; i < num_particles; i++) {
    beta += un_double_dist(gen) + 2 * beta;
    while(beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1)%num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
  for(auto& particle : particles) {
    particle.weight = 1.0;
  }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
