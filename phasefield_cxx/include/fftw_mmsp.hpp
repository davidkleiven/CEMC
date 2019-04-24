#ifndef FFTW_MMSP_H
#define FFTW_MMSP_H
#include "MMSP.grid.h"
#include "MMSP.vector.h"
#include "tools.hpp"
#include <vector>
#include <omp.h>
#include <iostream>
#ifdef HAS_FFTW
    #include <complex>
    #include <fftw3.h>
#else
    #include "fftw_complex_placeholder.hpp"
#endif

template<int dim>
using ft_grid_t = MMSP::grid<dim, MMSP::vector<fftw_complex> >;

class FFTW{
    public:
        enum class ExportType{REAL, IMAG, MODULUS};

        FFTW(unsigned int dim, const int *dims);
        ~FFTW();

        template<int dim>
        void execute(const ft_grid_t<dim> & grid_in, ft_grid_t<dim> &grid_out, int direction,
                     const std::vector<int> &ft_fields);
        
        void execute(const std::vector<double> &vec, std::vector<fftw_complex> &out, int dir);

        template<int dim>
        void execute(const MMSP::grid<dim, fftw_complex> &grid_in, MMSP::grid<dim, fftw_complex> &grid_out, int direction);

        /** Export buffer */
        void save_buffer(const std::string &fname, ExportType exp) const;
    private:
        unsigned int dimension{0};
        unsigned int num_elements_from_dims{0};
        fftw_complex *buffer{nullptr};
        fftw_plan forward_plan;
        fftw_plan backward_plan;

        template<int dim>
        void check_grids(const ft_grid_t<dim> & grid_in, ft_grid_t<dim> &grid_out) const;
        static bool multithread_initialized;
};


// Implement template methods
template<int dim>
void FFTW::check_grids(const ft_grid_t<dim> & grid_in, ft_grid_t<dim> &grid_out) const{
    int num_elements = MMSP::nodes(grid_in);

    // Ensure consistent nodes
    if (MMSP::nodes(grid_out) != num_elements){
        std::stringstream ss;
        ss << "Output and input lattice has different number of nodes! ";
        ss << "Input: " << num_elements << " Output: " << MMSP::nodes(grid_out);
        throw std::invalid_argument(ss.str());
    }

    // Ensure consistent number of fields
    if (MMSP::fields(grid_in) != MMSP::fields(grid_out)){
        std::stringstream ss;
        ss << "Number of fields in input and output lattice are different! ";
        ss << "Input: " << MMSP::fields(grid_in) << " Output: " << MMSP::fields(grid_out);
        throw std::invalid_argument(ss.str());
    }

    // Check consistent dimensions
    if (num_elements_from_dims != num_elements){
        std::stringstream ss;
        ss << "Dimension passed is inconsistent. Number of elements ";
        ss << "in the lattices " << num_elements << " Number of elements ";
        ss << "from dimension array: " << num_elements_from_dims;
        throw std::invalid_argument(ss.str());
    }
}

template<int dim>
void FFTW::execute(const ft_grid_t<dim> & grid_in, ft_grid_t<dim> &grid_out, int direction,
                   const std::vector<int> &ft_fields)
{
    #ifdef HAS_FFTW
        check_grids(grid_in, grid_out);

        double normalization = 1.0;

        if (direction == FFTW_BACKWARD){
            normalization = MMSP::nodes(grid_in);
        }

        // Loop over all fields that should be fourier transformed
        for (auto field : ft_fields){
            #ifndef NO_PHASEFIELD_PARALLEL
            #pragma omp parallel for
            #endif
            for (unsigned int i=0;i<MMSP::nodes(grid_in);i++){
                real(buffer[i]) = real(grid_in(i)[field]);
                imag(buffer[i]) = imag(grid_in(i)[field]);
            }
            // Perform the FFT
            // TODO: See if FFTW can utilize multithreading
            if (direction == FFTW_FORWARD){
                fftw_execute(forward_plan);
            }
            else{
                fftw_execute(backward_plan);
            }
        
            // Insert FT in the out field variable
            #ifndef NO_PHASEFIELD_PARALLEL
            #pragma omp parallel for
            #endif
            for (unsigned int i=0;i<MMSP::nodes(grid_out);i++){
                real(buffer[i]) /= normalization;
                imag(buffer[i]) /= normalization;

                real(grid_out(i)[field]) = real(buffer[i]);
                imag(grid_out(i)[field]) = imag(buffer[i]);
            }
        }
    #endif
};

template<int dim>
void FFTW::execute(const MMSP::grid<dim, fftw_complex> &grid_in, MMSP::grid<dim, fftw_complex> &grid_out, int direction){
    #ifdef HAS_FFTW
    double normalization = 1.0;

        if (direction == FFTW_BACKWARD){
            normalization = MMSP::nodes(grid_in);
        }

        // Loop over all fields that should be fourier transformed
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<MMSP::nodes(grid_in);i++){
            buffer[i] = grid_in(i);
        }

        // Perform the FFT
        // TODO: See if FFTW can utilize multithreading
        if (direction == FFTW_FORWARD){
            fftw_execute(forward_plan);
        }
        else{
            fftw_execute(backward_plan);
        }
    
        // Insert FT in the out field variable
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<MMSP::nodes(grid_out);i++){
            real(buffer[i]) /= normalization;
            imag(buffer[i]) /= normalization;
            grid_out(i) = buffer[i];
        }
    #endif
};
#endif