#include "fftw_mmsp.hpp"
#include <stdexcept>
using namespace std;

FFTW::FFTW(unsigned int dim, const int *dims): dimension(dim){
    #ifdef HAS_FFTW
        forward_plan =  fftwnd_create_plan(dim, dims, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);
        backward_plan = fftwnd_create_plan(dim, dims, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);

        num_elements_from_dims = 1;
        for (unsigned int i=0;i<dimension;i++){
            num_elements_from_dims *= dims[i];
        }
        buffer = new fftw_complex[num_elements_from_dims];
    #else
        throw runtime_error("FFTW class cannot be initialized when the code has been compiled without the HAS_FFTW macro!");
    #endif
}

FFTW::~FFTW(){
    #ifdef HAS_FFTW
        fftwnd_destroy_plan(forward_plan);
        fftwnd_destroy_plan(backward_plan);
        delete [] buffer;
    #endif
}