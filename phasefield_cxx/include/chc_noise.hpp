#ifndef CHC_NOISE_H
#define CHC_NOISE_H

#include <vector>
#include <string>

#include "thermal_noise_generator.hpp"

template<int dim>
class CHCNoise: public ThermalNoiseGenerator{
public:
    CHCNoise(double dt, double mobility, double amplitude, unsigned int L);
    ~CHCNoise();

    /** Create noise */
    void create(std::vector<double> &noise) const;

    void set_timestep(double new_dt){dt = new_dt;};

    /** Store noise at MMSP grid */
    void noise2grid(const std::string &fname, const std::vector<double> &noise) const;
private:
    double mobility{0.0};
    double dt{1.0};
    double amplitude{0.0};
    unsigned int L{1};
    MMSP::grid<dim, int> *indexGrid{nullptr};

    void chc_noise(const std::vector<double> &gaussian_white, std::vector<double> &chc_noise) const;
};
#endif