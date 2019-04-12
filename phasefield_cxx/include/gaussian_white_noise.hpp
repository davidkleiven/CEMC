#ifndef GAUSSIAN_WHITE_NOISE_H
#define GAUSSIAN_WHITE_NOISE_H
#include "thermal_noise_generator.hpp"

class GaussianWhiteNoise: public ThermalNoiseGenerator{
public:
    GaussianWhiteNoise(double dt, double amplitude): ThermalNoiseGenerator(dt), amplitude(amplitude){};

    virtual void create(std::vector<double> &noise) const override;
private:
    double amplitude{1.0};
};
#endif