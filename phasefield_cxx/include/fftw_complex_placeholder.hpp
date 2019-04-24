#ifndef FFTW_COMPLEX_PLACEHOLDER_H
#define FFTW_COMPLEX_PLACEHOLDER_H

#ifndef HAS_FFTW
#include <complex>
//typedef complex_t fftw_complex;

struct complex_t{
    double re;
    double im;
};

//typedef complex_t fftw_complex;
typedef double fftw_complex[2];
typedef int fftw_direction;

const int FFTW_FORWARD = 1;
const int FFTW_BACKWARD = -1;
const int FFTW_ESTIMATE = 0x01;

// Dummy value for an fftw ndplan
struct fftwnd_plan{};
#endif
#endif