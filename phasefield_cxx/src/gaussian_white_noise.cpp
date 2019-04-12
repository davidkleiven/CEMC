#include "gaussian_white_noise.hpp"
#include <cmath>
#include <stdexcept>

void GaussianWhiteNoise::create(std::vector<double> &noise) const{
    if (noise.size() % 2 != 0){
        throw std::invalid_argument("Use an even number of nodes when using Gaussian White Noise!");
    }

    double pi = acos(-1.0);
    double amp = sqrt(amplitude/dt);

    for (unsigned int i=0;i<noise.size()/2;i++){
        double rand1 = static_cast<double>(rand())/RAND_MAX;
        double rand2 = static_cast<double>(rand())/RAND_MAX;

        double normal1 = sqrt(-2.0*log(rand1))*cos(2*pi*rand2);
        double normal2 = sqrt(-2.0*log(rand1))*sin(2*pi*rand2);
        noise[2*i] = amp*normal1;
        noise[2*i+1] = amp*normal2;
    }
}