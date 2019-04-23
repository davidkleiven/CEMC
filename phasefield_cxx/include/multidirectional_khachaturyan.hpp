#ifndef MULTIDIRECTIONAL_KHACKATURYAN_H
#define MULTIDIRECTIONAL_KHACKATURYAN_H
#include <map>
#include "tools.hpp"
#include <stdexcept>
#include "khachaturyan.hpp"
#include "multidirectional_khach_data_logger.hpp"

#ifdef HAS_FFTW
    #include <complex>
    #include <fftw.h>
    #include "fftw_mmsp.hpp"
#endif
#include "fftw_complex_placeholder.hpp"

class MultidirectionalKhachaturyan{
    public:
        MultidirectionalKhachaturyan(double max_order_param): max_order_param(max_order_param){};
        ~MultidirectionalKhachaturyan();

        /** Add a new model */
        void add_model(const Khachaturyan &model, unsigned int shape_field);

        /** Return the number of models */
        unsigned int num_models() const{return strain_models.size();};

        template<int dim>
        void functional_derivative(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_in, 
            MMSP::grid<dim, MMSP::vector<fftw_complex> >&grid_out, const std::vector<int> &shape_fields);

        double get_last_strain_energy() const{return last_strain_energy;};

        template<int dim>
        void set_logger(MultidirectionalKhachDataLogger<dim> &new_logger);
    private:
        FFTW *fft{nullptr};
        std::map<unsigned int, Khachaturyan> strain_models;
        double max_order_param{1.0};
        double last_strain_energy{0.0};

        template<int dim>
        double fourier_integral(const MMSP::grid<dim, MMSP::vector<fftw_complex> >&ft_fields, const std::vector<int> &shape_fields) const;

        template<int dim, class T>
        double misfit_contribution(const MMSP::grid<dim, MMSP::vector<T> > &fields, const std::vector<int> &shape_fields) const;

        void get_effective_stresses(std::vector<mat3x3> &eff_stress) const;
        void index_map(const std::vector<int> &shape_fields, std::map<unsigned int, unsigned int> &mapping) const;

        // Explicitly set loggers to avoid making the entire class a template class
        MultidirectionalKhachDataLogger<1> *logger1{nullptr};
        MultidirectionalKhachDataLogger<2> *logger2{nullptr};
        MultidirectionalKhachDataLogger<3> *logger3{nullptr};

        template<int dim>
        MultidirectionalKhachDataLogger<dim>* logger();
};



// Implementation of template functions
template<int dim>
void MultidirectionalKhachaturyan::functional_derivative(const MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_in,
    MMSP::grid<dim, MMSP::vector<fftw_complex> > &grid_out, const std::vector<int> &shape_fields){
        #ifndef HAS_FFTW
            throw std::runtime_error("The package was compiled without FFTW!");
        #endif

        if (logger<dim>() != nullptr){
            clean_up(*logger<dim>());
        }
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
            shape_squared(i)[field].re = pow(grid_in(i)[field].re/max_order_param, 2);
            shape_squared(i)[field].im = 0.0;
        }

        if (logger<dim>() != nullptr){
            logger<dim>()->shape_squared_in = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(shape_squared);
            logger<dim>()->shape_squared_in->copy(shape_squared);
        }

        double misfit_energy_contrib = misfit_contribution(shape_squared, shape_fields);
        
        // Perform the fourier transform of all fields
        fft->execute(shape_squared, grid_out, FFTW_FORWARD, shape_fields);

        if (logger<dim>() != nullptr){
            logger<dim>()->fourier_shape_squared = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(grid_out);
            logger<dim>()->fourier_shape_squared->copy(grid_out);
        }

        double shape_contrib = fourier_integral(grid_out, shape_fields);
        last_strain_energy = misfit_energy_contrib - shape_contrib;

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

        if (logger<dim>() != nullptr){
            logger<dim>()->eff_stresses = eff_stresses;
            logger<dim>()->b_tensor_indx = b_tensor_indx;
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
            
            bool is_origin = false;
            if (abs(k) < 1E-6){
                //continue;
                is_origin = true;
            }
            else{
                divide(k_vec, k); // Convert to unit vector;
            }

            double *unit_vec_raw_ptr = &(k_vec[0]);
            mat3x3 G;
            mat3x3 B_tensor;
            strain_models.begin()->second.green_function(G, unit_vec_raw_ptr);

            if (is_origin){
                for (unsigned int i=0;i<3;i++)
                for (unsigned int j=0;j<3;j++){
                    B_tensor[i][j] = 0.0;
                }
            }
            else{
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

                    int indx2 = b_tensor_indx[field2];
                    int field_indx2 = shape_fields[field2];
                    // temp_grid(i)[field_indx1].re += B_tensor[indx][indx]*grid_out(i)[field_indx2].re;
                    // temp_grid2(i)[field_indx1].re += misfit_energy[indx][indx]*shape_squared(i)[field_indx2].re;
                    // temp_grid(i)[field_indx1].im += B_tensor[indx][indx]*grid_out(i)[field_indx2].im;
                    // temp_grid2(i)[field_indx1].im += misfit_energy[indx][indx]*shape_squared(i)[field_indx2].im;

                    temp_grid(i)[field_indx1].re += B_tensor[indx][indx2]*grid_out(i)[field_indx2].re;
                    temp_grid2(i)[field_indx1].re += misfit_energy[indx][indx2]*shape_squared(i)[field_indx2].re;
                    temp_grid(i)[field_indx1].im += B_tensor[indx][indx2]*grid_out(i)[field_indx2].im;
                    temp_grid2(i)[field_indx1].im += misfit_energy[indx][indx2]*shape_squared(i)[field_indx2].im;
                }
            }
        }

        // Update the origin by averaging the neighbours
        for (auto field : shape_fields){
            fftw_complex avg = average_nearest_neighbours(temp_grid, field, 0);
            temp_grid(0)[field] = avg;
        }
        

        if (logger<dim>() != nullptr){
            logger<dim>()->b_tensor_dot_ft_squared = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(temp_grid);
            logger<dim>()->b_tensor_dot_ft_squared->copy(temp_grid);
            logger<dim>()->misfit_energy_contrib = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(temp_grid2);
            logger<dim>()->misfit_energy_contrib->copy(temp_grid2);
        }

        // Inverse fourier transform
        fft->execute(temp_grid, grid_out, FFTW_BACKWARD, shape_fields);

        // Calculate the functional derivative (re-use temp_grid)
        for (unsigned int i=0;i<MMSP::nodes(temp_grid);i++)
        for (auto field : shape_fields){
            // Grid in is real, grid out should be real
            temp_grid(i)[field].re = 2*(grid_in(i)[field].re/max_order_param)*(temp_grid2(i)[field].re - grid_out(i)[field].re);
            temp_grid(i)[field].im = 0.0; // temp_grid should be real at this stage
        }

        // Not nessecary if the next line is used
        temp_grid.swap(grid_out);

        // Perform forward fourier transform again
        //fft_mmsp_grid(temp_grid, grid_out, FFTW_FORWARD, dims, shape_fields);
    }

    template<int dim>
    double MultidirectionalKhachaturyan::fourier_integral(const MMSP::grid<dim, MMSP::vector<fftw_complex> > &ft_fields, const std::vector<int> &shape_fields) const{
        double integral = 0.0;

        std::vector<mat3x3> eff_stress;
        get_effective_stresses(eff_stress);

        std::map<unsigned int, unsigned int> field2tensor_indx;
        index_map(shape_fields, field2tensor_indx);

        MMSP::vector<double> k_vec(3);
        for (unsigned int i=0;i<3;i++){
            k_vec[i] = 0.0;
        }

        mat3x3 green;

        for (int field1 : shape_fields)
        for (int field2 : shape_fields){
            unsigned int b_indx1 = field2tensor_indx.at(field1);
            unsigned int b_indx2 = field2tensor_indx.at(field2);

            double stress_strain_contracted = 0.0;
            for (unsigned int i=0;i<3;i++)
            for (unsigned int j=0;j<3;j++){
                stress_strain_contracted += eff_stress[b_indx1][i][j]*strain_models.at(field2).get_misfit()[i][j];
            }

            #ifndef NO_PHASEFIELD_PARALLEL
            #pragma omp parallel for reduction(+ : integral)
            #endif
            for (int node=0;node<MMSP::nodes(ft_fields);node++){
                MMSP::vector<int> pos = ft_fields.position(node);

                // Convert position to k-vector
                k_vector(pos, k_vec, MMSP::xlength(ft_fields));

                // Calculate the green function
                double k = norm(k_vec);
                

                if (abs(k) < 1E-6){
                    continue;
                }

                divide(k_vec, k); // Convert to unit vector
                const Khachaturyan& khac_obj = strain_models.begin()->second;
                khac_obj.green_function(green, &(k_vec[0]));

                // Construct B_tensor element
                double element = B_tensor_element(k_vec, green, eff_stress[b_indx1], eff_stress[b_indx2]);
                integral += element*(ft_fields(node)[field1].re*ft_fields(node)[field2].re + \
                    ft_fields(node)[field1].im*ft_fields(node)[field2].im);
            }
        }
        return 0.5*integral;
    }

    template<int dim, class T>
    double MultidirectionalKhachaturyan::misfit_contribution(const MMSP::grid<dim, MMSP::vector<T> > &fields, const std::vector<int> &shape_fields) const{
        double integral = 0.0;

        std::vector<mat3x3> eff_stress;
        get_effective_stresses(eff_stress);

        std::map<unsigned int, unsigned int> field2tensor;
        index_map(shape_fields, field2tensor);

        for (int field1 : shape_fields)
        for (int field2 : shape_fields){
            unsigned int tensor_indx1 = field2tensor.at(field1);
            unsigned int tensor_indx2 = field2tensor.at(field2);

            double factor = 0.0;
            for (unsigned int i=0;i<3;i++)
            for (unsigned int j=0;j<3;j++){
                factor += eff_stress[tensor_indx1][i][j]*strain_models.at(field2).get_misfit()[i][j];
            }

            #ifndef NO_PHASEFIELD_PARALLEL
            #pragma omp parallel for reduction(+ : integral)
            #endif
            for (int node=0;node<MMSP::nodes(fields);node++){
                integral += factor*real_field(fields(node)[field1])*real_field(fields(node)[field2]);
            }
        }
        return 0.5*integral;
    }


    // Get logger
    template<>
    MultidirectionalKhachDataLogger<1> *MultidirectionalKhachaturyan::logger(){return logger1;};

    template<>
    MultidirectionalKhachDataLogger<2> *MultidirectionalKhachaturyan::logger(){return logger2;};

    template<>
    MultidirectionalKhachDataLogger<3> *MultidirectionalKhachaturyan::logger(){return logger3;};

    // Set logger
    template<>
    void MultidirectionalKhachaturyan::set_logger(MultidirectionalKhachDataLogger<1> &new_logger){logger1 = &new_logger;};

    template<>
    void MultidirectionalKhachaturyan::set_logger(MultidirectionalKhachDataLogger<2> &new_logger){logger2 = &new_logger;};

    template<>
    void MultidirectionalKhachaturyan::set_logger(MultidirectionalKhachDataLogger<3> &new_logger){logger3 = &new_logger;};
#endif