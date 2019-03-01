#ifndef PHASEFIELD_TOOLS_H
#define PHASEFIELD_TOOLS_H
#include "MMSP.grid.h"
#include "MMSP.vector.h"
#include "fftw_complex_placeholder.hpp"
#include <vector>

template<int dim, typename T>
T partial_double_derivative(const MMSP::grid<dim, T> &GRID, const MMSP::vector<int> &x, unsigned int dir){
    MMSP::vector<int> s = x;
    const T& y = GRID(x);
    s[dir] += 1;
    const T& yh = GRID(s);
    s[dir] -= 2;
    const T& yl = GRID(s);
    s[dir] += 1;

    double weight = 1.0/pow(dx(GRID, dir), 2.0);
    return weight*(yh - 2.0*y + yl);
}

template<int dim, typename T>
T partial_double_derivative(const MMSP::grid<dim, T> &GRID, unsigned int node_index, unsigned int dir){
    MMSP::vector<int> x = GRID.position(node_index);
    return partial_double_derivative(GRID, x, dir);
}

#ifdef HAS_FFTW
    #include <complex>
    #include <fftw.h>
    template<int dim>
    void fft_mmsp_grid(const MMSP::grid<dim, MMSP::vector<fftw_complex> > & grid_in, MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, fftw_direction direction,
                    const int *dims, const std::vector<int> &ft_fields);
#endif

void k_vector(const MMSP::vector<int> &pos, MMSP::vector<double> &k_vec, int N);

template<int dim>
void get_dims(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_in, int dims[3]);

double norm(const MMSP::vector<double> &vec);
void dot(const mat3x3 &mat1, const MMSP::vector<double> &vec, MMSP::vector<double> &out);
double dot(const MMSP::vector<double> &vec1, const MMSP::vector<double> &vec2);

template<class T>
void divide(MMSP::vector<T> &vec, double factor);

#include "tools.tpp"
#endif