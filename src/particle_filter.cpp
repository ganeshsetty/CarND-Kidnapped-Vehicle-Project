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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	//Tweak
	num_particles = 100;
	
	weights.resize(num_particles,1.0);
	default_random_engine gen;
	 
	// Create normal (Gaussian) distributions for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles
	for (int i = 0; i < num_particles; i++) {
		
		Particle new_particle;
		new_particle.id = i;
		new_particle.x = dist_x(gen);
		new_particle.y = dist_y(gen);
		new_particle.theta = dist_theta(gen);
		new_particle.weight = 1.0;
        
        // Add initialized new_particle  to list of particles
        particles.push_back(new_particle);
	}
    
	// Once all particles are initialized set the flag as true
	is_initialized = true;

}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	double v_yawdot,v_dt;
	default_random_engine gen;
	
	//Calculated once and stored
	double yawdot_dt = yaw_rate*delta_t;
		
	// Create normal (Gaussian) distributions for noises 
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);
	
	if(fabs(yaw_rate) > 0.001) {
		v_yawdot = velocity/yaw_rate;
		
		for (unsigned int i = 0; i < num_particles ; i++) {
			particles[i].x +=  (v_yawdot)*(sin(particles[i].theta + yawdot_dt) - sin(particles[i].theta)) + noise_x(gen); //noise added
			particles[i].y +=  (v_yawdot)*(cos(particles[i].theta) - cos(particles[i].theta + yawdot_dt)) + noise_y(gen);
			particles[i].theta += yawdot_dt + noise_theta(gen);
		}
	}
	else {
		v_dt = velocity * delta_t;
		
		for (unsigned i = 0; i < num_particles ; i++) {
			particles[i].x +=  (v_dt * cos(particles[i].theta)) + noise_x(gen);
			particles[i].y +=  (v_dt * sin(particles[i].theta)) + noise_y(gen);
			particles[i].theta +=  yawdot_dt + noise_theta(gen); // yawdot_dt can be omitted as yaw_rate is very small
		}
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
	
	
	//observation measurements-It is in local/vehicle co-ordinates space(x,y)	
	LandmarkObs obs; 
	
	//Used in multi-variate Gaussian probability calculation 
	double var_x = pow(std_landmark[0],2);
	double var_y = pow(std_landmark[1],2);
	double covar_xy = std_landmark[0] * std_landmark[1];
	
	double weights_sum = 0;
	
	// Calculate particles final weights
	for (unsigned int i = 0; i < num_particles; i++)
	{
        //Initialize weight as 1 as it is multiplication factor in calculation of particle's final weight
		double weight = 1.0;		
		for (unsigned int j = 0; j < observations.size() ; j++)
		{	
			obs = observations[j];
			
			//Predicted measurement x,y
			//Transform the observations (in local co-ordinates) for each particle (in global co-ordinates) i.e, to map co-ordinates space 
			double predicted_x = obs.x * cos(particles[i].theta) - obs.y * sin(particles[i].theta) + particles[i].x;
			double predicted_y = obs.x * sin(particles[i].theta) + obs.y * cos(particles[i].theta) + particles[i].y;
		
			//Associate the closest landmark to each transformed observation
		    Map::single_landmark_s closest_landmark;	
            
			//Accounting for only within sensor range
			double min_distance = sensor_range;
			
			//from list of landmarks in the map, choose the  landmark which is closest to predicted measurement
			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
			{
				Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
				
				//calculate Euclidean distance between predicted measurement and landmark
				double distance = dist( landmark.x_f, landmark.y_f,predicted_x, predicted_y);
				
				// search for landmarks within sensor range and choose one closest to predicted measurement
				if(distance < min_distance)
				{
					min_distance = distance;
					closest_landmark = landmark;
				}
			} //end of landmark list loop
			
			//Calculating each predicted measurement's Multivariate-Gaussian probability
			double x_diff = predicted_x - closest_landmark.x_f;
			double y_diff = predicted_y - closest_landmark.y_f;
									
			double expr_num = exp(-((x_diff * x_diff)/(2*var_x) + (y_diff * y_diff)/(2*var_y)));
			double expr_denom = 2*M_PI*covar_xy;
			
			//particle's final weight will be calculated as the product of each predicted measurement's Multivariate-Gaussian probability
			weight *= expr_num/expr_denom;
		} // end of observation loop
		
		//Store particle's final weight
	    particles[i].weight = weight;	
        weights[i] = weight;
	} // end of particles loop
	
	// Normalizing weights
	for (auto n : weights) 
        weights_sum += n;
    
    for (unsigned int i = 0; i << num_particles ; i++) 	{
		particles[i].weight /= weights_sum;
		weights[i] /= weights_sum;
	}
		
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	discrete_distribution<int> weights_dist(weights.begin(), weights.end());
	vector<Particle> new_particles;
	
	default_random_engine gen;
	// resample particles
    for (unsigned int i = 0; i < num_particles; ++i)
        new_particles.push_back(particles[weights_dist(gen)]);
	
    //new set of particles(resampled)
	particles = new_particles;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
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
