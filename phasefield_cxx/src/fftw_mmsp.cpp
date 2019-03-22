#include "fftw_mmsp.hpp"
#include <stdexcept>
#include <fstream>
#include <omp.h>
using namespace std;

bool FFTW::multithread_initialized = false;

FFTW::FFTW(unsigned int dim, const int *dims): dimension(dim){
    #ifdef HAS_FFTW
        if (!this->multithread_initialized){
            #ifndef NO_PHASEFIELD_PARALLEL
                int return_code = fftw_init_threads();
                if (return_code == 0){
                    throw runtime_error("Could not initialize multithreaded FFTW!");
                }
            #endif
            this->multithread_initialized = true;
        }

        #ifndef NO_PHASEFIELD_PARALLEL
            fftw_plan_with_nthreads(omp_get_max_threads());
        #endif
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

void FFTW::save_buffer(const string &fname, ExportType exp) const{
    ofstream ofs(fname);

    if (!ofs.good()){
        throw runtime_error("Could not write buffer to file!");
    }

    for (unsigned int i=0;i<num_elements_from_dims;i++){
        switch(exp){
            case ExportType::IMAG:
                ofs << buffer[i].im << ",";
                break;
            case ExportType::REAL:
                ofs << buffer[i].re << ",";
                break;
            case ExportType::MODULUS:
                ofs << sqrt(pow(buffer[i].re, 2) + pow(buffer[i].im, 2)) << ",";
                break;
        }
    }
    ofs.close();
}