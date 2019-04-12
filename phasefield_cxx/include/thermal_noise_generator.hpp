#ifndef THERMAL_NOISE_GENERATOR_H
#define THERMAL_NOISE_GENERATOR_H
#include <vector>

class ThermalNoiseGenerator{
public:
    ThermalNoiseGenerator(double dt): dt(dt){};

    virtual void create(std::vector<double> &noise) const = 0;

    void set_timestep(double new_dt){dt = new_dt;};
protected:
    double dt{1.0};
};
#endif