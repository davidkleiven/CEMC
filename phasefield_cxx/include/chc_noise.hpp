#ifndef CHC_NOISE_H
#define CHC_NOISE_H

#include <vector>

template<int dim>
class CHCNoise{
public:
    CHCNoise(double mobility, double dt, double amplitude, unsigned int L);
    ~CHCNoise();

    /** Create noise */
    void create(std::vector<double> &noise) const;

    void set_timestep(double new_dt){dt = new_dt;};
private:
    double mobility{0.0};
    double dt{1.0};
    double amplitude{0.0};
    unsigned int L{1};
    MMSP::grid<dim, int> *indexGrid{nullptr};

    void chc_noise(const std::vector<double> &gaussian_white, std::vector<double> &chc_noise) const;
};
#endif