/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::normal_distribution;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // Set the number of particles

  // Creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for(int i = 0; i < num_particles; i++){
    double ini_x = dist_x(gen);
    double ini_y = dist_y(gen);
    double ini_theta = dist_theta(gen);
	// Create a new particle    
    Particle p = {i, ini_x, ini_y, ini_theta, 1.0};
    // Add the particle to the vector
  	particles.push_back(p);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  if(yaw_rate < 0.0001)
    yaw_rate = 0.0001;
  // Use the bicycle model from the Udacity class
  for (Particle &p : particles){
    double x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
    double y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    double theta = p.theta + yaw_rate * delta_t;
    
    // add noise with a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);
    
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // Use nearest neighbour to associate
  //Be careful about the index of the landmarks and assoicated observations
  vector<LandmarkObs> associatedLandmarks;
  
  for(LandmarkObs pre : predicted){
    
    double closest_dist = INFINITY;
    LandmarkObs associatedLandmark;
    // Obtain the shortest distance and the observation map coordinates
    for(LandmarkObs &obs : observations){
      double distance = dist(pre.x, pre.y, obs.x, obs.y);
      if(distance < closest_dist){
        closest_dist = distance;
        associatedLandmark.id = pre.id;
        associatedLandmark.x = obs.x;
        associatedLandmark.y = obs.y;
      }        
    }
    associatedLandmarks.push_back(associatedLandmark);
  }
  // the observations in the order of predicted landmarks
  observations = associatedLandmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (Particle &p : particles){
//     std::cout << "particle weight before update: " << p.weight << std::endl;
    // Transform the vehicle's coordinte to map's coordinate
    vector<LandmarkObs> tObservations;
    for(LandmarkObs obs : observations){
      LandmarkObs tObs = {obs.id, 
                          p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta), 
                          p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta)
                          }; 
      tObservations.push_back(tObs);
    }
    
    // Find the landmarks within the range of sensor
    vector<LandmarkObs> predicted;
//     std::cout << "number of landmarks: " << map_landmarks.landmark_list.size() << std::endl;
    for(Map::single_landmark_s slm : map_landmarks.landmark_list){
      double distToCar = dist(slm.x_f, slm.y_f, p.x, p.y);
      // check if the landmark is within the sensor range
      if(distToCar < sensor_range){
        LandmarkObs pre = {slm.id_i, slm.x_f, slm.y_f};
        predicted.push_back(pre);
      }
    }
    
    // associate landmarks with observations transformed map coordinates
    dataAssociation(predicted, tObservations);
    
    // set association for debugging
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y; 
    for(int i=0; i<tObservations.size(); i++){
      associations.push_back(tObservations.at(i).id);
      sense_x.push_back(tObservations.at(i).x);
      sense_y.push_back(tObservations.at(i).y);
    }
    SetAssociations(p, associations, sense_x, sense_y);
    
    // Calculate the weight of the current observation
    for(int i=0; i<tObservations.size(); i++){
      double weight = multiv_prob(std_landmark[0], std_landmark[1], tObservations.at(i).x, tObservations.at(i).y, 
                                  predicted.at(i).x, predicted.at(i).y);
      p.weight = p.weight * weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> resampleParticles;
  // Get the vector of paticle weight
  vector<double> weights;
  for(Particle p : particles){
//     std::cout << "particle weight after update: " << p.weight << std::endl;
    weights.push_back(p.weight);
  }  
  // Resample use the descrete distribution, take index of particle according the weight
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  for(int i = 0; i<num_particles; i++){
    int index = d(gen);
    resampleParticles.push_back(particles[index]);  
  }
  particles = resampleParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}