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

#define EPS 1e-4
#define NUM_PARTICLES 100

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = NUM_PARTICLES;
  particles.resize(NUM_PARTICLES);
  // normal distributions
  std::default_random_engine gen;
  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);
  // generate particles with normal distribution with mean on GPS values
  for (auto& p : particles) {
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1.0;
  }
  is_initialized = true;
}

// landmark prediction.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::default_random_engine gen;
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);

  for(auto& particle : particles) {
    double theta = particle.theta;

    if (fabs(yaw_rate) < EPS) {
      particle.x += velocity * delta_t * cos(theta);
      particle.y += velocity * delta_t * sin(theta);
    }
    else {
      particle.x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
      particle.y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
      particle.theta += yaw_rate * delta_t;
    }
    // noise
    particle.x += N_x(gen);
    particle.y += N_y(gen);
    particle.theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
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

// reassign weights to each particles.
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // There are 4 steps in total. Picking landmark within sensor range, transforming coordinates to map, finding nearlest prediction of landmark for each transformed coordinate, and assign weight to particles.
    for(auto& particle : particles) {
      double particle_x = particle.x;
      double particle_y = particle.y;
      double particle_theta = particle.theta;
      vector<LandmarkObs> p_landmarks;
      particle.weight = 1.0;
      
      // step 1:
      // Exclude uninterest landmark, only pick up landmark within sensor range 
      for(const auto& m_landmark : map_landmarks.landmark_list) {
        float lm_x = m_landmark.x_f;
        float lm_y = m_landmark.y_f;
        int lm_id = m_landmark.id_i;
        // only consider landmarks within sensor range of the particle
        if(fabs(lm_x - particle_x) <= sensor_range && fabs(lm_y - particle_y) <= sensor_range) {
          p_landmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y});
        }
      }

      // step 2:
      // Transform coordinates of observations from world onto map
      // The theory infer to Homogenous, here is usefull learning resource: https://www.youtube.com/watch?v=tAu8-gkxAcE
      vector<LandmarkObs> transform_observations;
      for(const auto& obs : observations) {
        double t_x = particle_x + cos(particle_theta) * obs.x - sin(particle_theta) * obs.y;
        double t_y = particle_y + sin(particle_theta) * obs.x + cos(particle_theta) * obs.y;
        transform_observations.push_back(LandmarkObs{obs.id, t_x, t_y});
      }

      // step 3:
      // find out nearlest prediction to each of transform_observations
      dataAssociation(p_landmarks, transform_observations);
      
      // step 4:
      // Calculate weight for particle connecting to landmark nearby transformed observation.
      for(const auto& t_obs : transform_observations) {
        double o_x, o_y, p_x, p_y;
        o_x = t_obs.x;
        o_y = t_obs.y;
        int asso_predict_id = t_obs.id;
        
        for(const auto& p_landmark : p_landmarks) {
          if(p_landmark.id == asso_predict_id) {
            p_x = p_landmark.x;
            p_y = p_landmark.y;
            break;
          }
        }
        // weight for this observation with multivariate Gaussian
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];
          
        double mep_x = pow(o_x-p_x, 2) / (2*pow(std_x, 2));
        double mep_y = pow(o_y-p_y, 2) / (2*pow(std_y, 2));

        double obs_w = (1.0/(2*M_PI*std_x*std_y)) * exp(-(mep_x + mep_y));
        particle.weight *= obs_w;
      }
    } // end particles

    // normalization
    /*
    double norm_factor = 0.0;
    for(const auto& particle : particles) {
      norm_factor += particle.weight;
    }
    for(auto& particle : particles) {
      particle.weight /= (norm_factor + numeric_limits<double>::epsilon());
    }
    */
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  // Basic theory
  /*
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for(unsigned int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }

  std::default_random_engine gen;
  std::uniform_real_distribution<double> un_double_dist(0.0, max_weight);
  std::uniform_int_distribution<int> un_int_dist(0, num_particles-1);
  int index = un_int_dist(gen);
  double beta = 0.0;
  vector<Particle> resampledParticles;
  for(unsigned int i = 0; i < num_particles; i++) {
    beta += 2*un_double_dist(gen);
    while(beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1)%num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
  */
  // Alternated purpose
  vector<double> weights;
  weights.resize(num_particles);
  for(unsigned int i = 0; i < num_particles; i++) {
    weights[i] = particles[i].weight;
  }
  std::random_device rd;
  std::mt19937 mt_gen_19937(rd());
  std::discrete_distribution<> ddist(weights.begin(), weights.end());

  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);
  for(unsigned int i = 0; i < num_particles; i++) {
    int index = ddist(mt_gen_19937);
    resampled_particles[i] = particles[index];
  }
  particles = resampled_particles;
  weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
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
