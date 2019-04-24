#include "fftw_mmsp.hpp"
#include <stdexcept>
#include <fstream>
#include <omp.h>
using namespace std;

bool FFTW::multithread_initialized = false;

FFTW::FFTW(unsigned int dim, const int *dims): dimension(dim){
    #ifdef HAS_FFTW
        // TODO: Make this work with hyperthreading
        
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
        //buffer = new fftw_complex[num_elements_from_dims];
        buffer = fftw_alloc_complex(num_elements_from_dims);

        num_elements_from_dims = 1;
        for (unsigned int i=0;i<dimension;i++){
            num_elements_from_dims *= dims[i];
        }

        forward_plan =  fftw_plan_dft(dim, dims, buffer, buffer, FFTW_FORWARD, FFTW_ESTIMATE);
        backward_plan = fftw_plan_dft(dim, dims, buffer, buffer, FFTW_BACKWARD, FFTW_ESTIMATE);

    // #else
    //     throw runtime_error("FFTW class cannot be initialized when the code has been compiled without the HAS_FFTW macro!");
    #endif
}

FFTW::~FFTW(){
    #ifdef HAS_FFTW
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        fftw_free(buffer);
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
                ofs << imag(buffer[i]) << ",";
                break;
            case ExportType::REAL:
                ofs << real(buffer[i]) << ",";
                break;
            case ExportType::MODULUS:
                ofs << sqrt(pow(real(buffer[i]), 2) + pow(imag(buffer[i]), 2)) << ",";
                break;
        }
    }
    ofs.close();
}

void FFTW::execute(const vector<double> &vec, vector<fftw_complex> &out, int direction){

    if (vec.size() != num_elements_from_dims){
        stringstream ss;
        ss << "The length of the vector must match! ";
        ss << "Got " << vec.size() << ". Expected: " << num_elements_from_dims;
        throw invalid_argument(ss.str());
    }

    // Transfer to buffer
    for (unsigned int i=0;i<vec.size();i++){
        real(buffer[i]) = vec[i];
        imag(buffer[i]) = 0.0;
    }

    double normalization = 1.0;
    if (direction == FFTW_BACKWARD){
        normalization = vec.size();
    }

    if (direction == FFTW_FORWARD){
        fftw_execute(forward_plan);
    }
    else{
        fftw_execute(backward_plan);
    }

    // Transfer back
    for (unsigned int i=0;i<vec.size();i++){
        real(out[i]) = real(buffer[i])/normalization;
        imag(out[i]) = imag(buffer[i])/normalization;
    }
}