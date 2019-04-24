#include "tools.hpp"

template<int dim>
void strain(const Khachaturyan &khac, const MMSP::grid<dim, fftw_complex> &shape, MMSP::grid<dim, MMSP::vector<double> > &strain, FFTW &fft){

    MMSP::grid<dim, fftw_complex> &ft_shape(shape);
    fft.execute(shape, ft_shape, FFTW_FORWARD);

    MMSP::vector<double> k_vec(3);
    mat3x3 eff_stress;
    mat3x3 green;
    khac.effective_stress(eff_stress);

    for (unsigned int i=0;i<3;i++){
        k_vec[i] = 0.0;
    }

    
    for (unsigned int dir=0;dir<dim;dir++){
        MMSP::grid<dim, fftw_complex> ft_displacement;

        for (unsigned int node=0;node<MMSP::nodes(shape);node++){
            MMSP::vector<int> pos = shape.position(node);
            k_vector(pos, k_vec, dim);
            double k = norm(k_vec);

            if (abs(k) < 1E-6){
                continue;
            }

            divide(k_vec, k);
            khac.green_function(green, &k_vec[0]);

            fftw_complex force[3];
            generalized_force(eff_stress, k_vec, ft_shape(node), force);

            real(ft_displacement(node)) = 0.0;
            imag(ft_displacement(node)) = 0.0;

            for (unsigned int j=0;j<3;j++){
                real(ft_displacement(dir)) += green[dir][j]*real(force[j]);
                imag(ft_displacement(dir)) += green[dir][j]*imag(force[j]);
            }
        }

        MMSP::grid<dim, fftw_complex> ft_disp_cpy(ft_displacement);

        // Calculate the strains
        for (unsigned int kdir=0;kdir<dim;kdir++){
            for (unsigned int node=0;node<MMSP::nodes(ft_displacement);node++){
                MMSP::vector<int> pos = shape.position(node);
                k_vector(pos, k_vec, dim);
                double k = norm(k_vec);
                if (abs(k) < 1E-6){
                    continue;
                }
                divide(k_vec, k);
                real(ft_disp_cpy(node)) = k_vec[kdir]*real(ft_displacement(node));
                imag(ft_disp_cpy(node)) = k_vec[kdir]*imag(ft_displacement(node));
            }

            MMSP::grid<dim, fftw_complex> strain_out(ft_disp_cpy);
            fft.execute(ft_disp_cpy, strain_out, FFTW_BACKWARD);

            unsigned int field = voigt_index(dir, kdir);
            for (unsigned int node=0;node<MMSP::nodes(ft_displacement);node++){
                strain(node)[field] += 0.5*real(strain_out(node));
            }
        }
    }
}