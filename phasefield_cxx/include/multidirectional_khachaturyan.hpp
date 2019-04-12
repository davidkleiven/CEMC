#ifndef MULTIDIRECTIONAL_KHACKATURYAN_H
#define MULTIDIRECTIONAL_KHACKATURYAN_H
#include <map>
#include "tools.hpp"
#include <stdexcept>
#include "khachaturyan.hpp"

#ifdef HAS_FFTW
    #include <complex>
    #include <fftw.h>
    #include "fftw_mmsp.hpp"
#endif
#include "fftw_complex_placeholder.hpp"

class MultidirectionalKhachaturyan{
    public:
        MultidirectionalKhachaturyan(){};
        ~MultidirectionalKhachaturyan();

        /** Add a new model */
        void add_model(const Khachaturyan &model, unsigned int shape_field);

        /** Return the number of models */
        unsigned int num_models() const{return strain_models.size();};

        template<int dim>
        void functional_derivative(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_in, 
            MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_out, const std::vector<int> &shape_fields);
    private:
        FFTW *fft{nullptr};
        std::map<unsigned int, Khachaturyan> strain_models;
        double B_tensor_element(MMSP::vector<double> &dir, const mat3x3 &green, const mat3x3 &eff_stress1, const mat3x3 &eff_stress2) const;
        double contract_tensors(const mat3x3 &mat1, const mat3x3 &mat2) const;
};



// Implementation of template functions
template<int dim>
void MultidirectionalKhachaturyan::functional_derivative(const MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_in,
    MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, const std::vector<int> &shape_fields){
        #ifndef HAS_FFTW
            throw std::runtime_error("The package was compiled without FFTW!");
        #endif
        //std::cout << strain_models[0].elastic.size() << std::endl;

        int dims[3];
        get_dims(grid_in, dims);

        MMSP::grid<dim, MMSP::vector<fftw_complex> > shape_squared(grid_in);

        if (fft == nullptr){
            fft = new FFTW(dim, dims);
        }

        // Calculate the square of the shape parameters
        for (unsigned int i=0;i<MMSP::nodes(grid_in);i++)
        for (auto field : shape_fields){
            shape_squared(i)[field].re = pow(grid_in(i)[field].re, 2);
            shape_squared(i)[field].im = 0.0;
        }
        
        // Perform the fourier transform of all fields
        fft->execute(shape_squared, grid_out, FFTW_FORWARD, shape_fields);

        MMSP::vector<double> k_vec(3);
        for (unsigned int i=0;i<3;i++){
            k_vec[i] = 0.0;
        }

        // Pre-calculate effective stresses
        std::map<unsigned int, mat3x3> eff_stresses;
        std::map<unsigned int, unsigned int> b_tensor_indx;
        int counter = 0;
    
        for (auto iter=strain_models.begin(); iter != strain_models.end();++iter){
            iter->second.effective_stress(eff_stresses[iter->first]);
            b_tensor_indx[iter->first] = counter++;
        }

        // Calculate the inner product between the effective stress and misfit strain
        mat3x3 misfit_energy;
        for (unsigned int field1=0;field1 < shape_fields.size();field1++)
        for (unsigned int field2=0;field2 < shape_fields.size();field2++){
            unsigned int indx = b_tensor_indx[field1];
            unsigned int indx2 = b_tensor_indx[field2];
            misfit_energy[indx][indx2] = contract_tensors(eff_stresses[shape_fields[field1]], strain_models[shape_fields[field2]].get_misfit());
        }
        

        MMSP::grid<dim, MMSP::vector<fftw_complex> > temp_grid(grid_in);
        MMSP::grid<dim, MMSP::vector<fftw_complex> > temp_grid2(grid_in);

        // Multiply with the Green function
        for (unsigned int i=0;i<MMSP::nodes(grid_out);i++){
            MMSP::vector<int> pos = grid_out.position(i);

            // Convert position to k-vector
            k_vector(pos, k_vec, MMSP::xlength(grid_in));

            // Calculate the green function
            double k = norm(k_vec);
            divide(k_vec, k); // Convert to unit vector
            double *unit_vec_raw_ptr = &(k_vec[0]);
            mat3x3 G;
            mat3x3 B_tensor;
            strain_models.begin()->second.green_function(G, unit_vec_raw_ptr);
            unsigned int row = 0;
            unsigned int col = 0;
            for (auto iter1=eff_stresses.begin(); iter1 != eff_stresses.end(); ++iter1){
                col = 0;
                for (auto iter2=eff_stresses.begin(); iter2 != eff_stresses.end(); ++iter2){
                    B_tensor[row][col] = B_tensor_element(k_vec, G, iter1->second, iter2->second);
                    col += 1;
                }
                row += 1;
            }

            // Update the shape fields
            for (unsigned int field1=0;field1<shape_fields.size();field1++){
                int indx = b_tensor_indx[field1];
                int field_indx1 = shape_fields[field1];
                temp_grid(i)[field_indx1].re = B_tensor[indx][indx]*grid_out(i)[field_indx1].re;
                temp_grid2(i)[field_indx1].re = misfit_energy[indx][indx]*shape_squared(i)[field_indx1].re;
                temp_grid(i)[field_indx1].im = B_tensor[indx][indx]*grid_out(i)[field_indx1].im;
                temp_grid2(i)[field_indx1].im = misfit_energy[indx][indx]*shape_squared(i)[field_indx1].im;
                for (unsigned int field2=0;field2<shape_fields.size();field2++){
                    if (field2 == field1){
                        continue;
                    }

                    int field_indx2 = shape_fields[field2];
                    temp_grid(i)[field_indx1].re += B_tensor[indx][indx]*grid_out(i)[field_indx2].re;
                    temp_grid2(i)[field_indx1].re += misfit_energy[indx][indx]*shape_squared(i)[field_indx2].re;
                    temp_grid(i)[field_indx1].im += B_tensor[indx][indx]*grid_out(i)[field_indx2].im;
                    temp_grid2(i)[field_indx1].im += misfit_energy[indx][indx]*shape_squared(i)[field_indx2].im;
                }
            }
        }

        // Inverse fourier transform
        fft->execute(temp_grid, grid_out, FFTW_BACKWARD, shape_fields);

        // Calculate the functional derivative (re-use temp_grid)
        for (unsigned int i=0;i<MMSP::nodes(temp_grid);i++)
        for (auto field : shape_fields){
            // Grid in is real, grid out should be real
            temp_grid(i)[field].re = 2*grid_in(i)[field].re*(temp_grid2(i)[field].re - grid_out(i)[field].re);
            temp_grid(i)[field].im = 0.0; // temp_grid should be real at this stage
        }

        // Not nessecary if the next line is used
        temp_grid.swap(grid_out);

        // Perform forward fourier transform again
        //fft_mmsp_grid(temp_grid, grid_out, FFTW_FORWARD, dims, shape_fields);
    }


#endif