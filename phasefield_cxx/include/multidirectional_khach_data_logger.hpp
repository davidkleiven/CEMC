#ifndef MULTIDIRECTIONAL_KHACH_DATA_LOGGER_H
#define MULTIDIRECTIONAL_KHACH_DATA_LOGGER_H

#include "MMSP.grid.h"
#include "MMSP.vector.h"

#include <map>
#include <array>

#ifdef HAS_FFTW
    #include <complex>
    #include <fftw.h>
#endif
#include "fftw_complex_placeholder.hpp"

typedef std::array< std::array<double, 3>, 3> mat3x3;

// Debug struct used to trace all intermediate results 
// during strain field calculation
template<int dim>
struct MultidirectionalKhachDataLogger{
    MMSP::grid<dim, MMSP::vector<fftw_complex> > *shape_squared_in{nullptr};
    MMSP::grid<dim, MMSP::vector<fftw_complex> > *fourier_shape_squared{nullptr};
    MMSP::grid<dim, MMSP::vector<fftw_complex> > *b_tensor_dot_ft_squared{nullptr};
    MMSP::grid<dim, MMSP::vector<fftw_complex> > *misfit_energy_contrib{nullptr};

    std::map<unsigned int, mat3x3> eff_stresses;
    std::map<unsigned int, unsigned int> b_tensor_indx;
    mat3x3 misfit_energy;
};

template<int dim>
void clean_up(MultidirectionalKhachDataLogger<dim> &logger){
    delete logger.shape_squared_in; logger.shape_squared_in = nullptr;
    delete logger.fourier_shape_squared; logger.fourier_shape_squared = nullptr;
    delete logger.b_tensor_dot_ft_squared; logger.b_tensor_dot_ft_squared = nullptr;
    delete logger.misfit_energy_contrib; logger.misfit_energy_contrib = nullptr;
};
#endif