#ifndef FFTW_COMPLEX_PLACEHOLDER_H
#define FFTW_COMPLEX_PLACEHOLDER_H

#ifndef HAS_FFTW
struct fftw_complex{
    double re;
    double im;
};
#endif
#endif