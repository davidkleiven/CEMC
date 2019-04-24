#ifndef ELASTICITY_UTIL_H
#define ELASTICITY_UTIL_H
#include "khachaturyan.hpp"
#include "MMSP.grid.h"
#include "fftw_mmsp.hpp"

#ifdef HAS_FFTW
    #include <complex>
    #include <fftw3.h>
#else
    #include "fftw_complex_placeholder.hpp"
#endif

template<int dim>
void strain(const Khachaturyan &khac, const MMSP::grid<dim, fftw_complex> &shape, MMSP::grid<dim, MMSP::vector<double> > &strain, FFTW &fft);

void generalized_force(const mat3x3 &eff_stress, const double reciprocal_dir[3], const fftw_complex &shape_func, double force[3]);

unsigned int voigt_index(unsigned int i, unsigned int j);
#include "elasticity_util.tpp"
#endif