#include "tools.hpp"

template<int dim>
void strain(const Khachaturyan &khac, const MMSP::grid<dim, fftw_complex> &shape, MMSP::grid<dim, MMSP::vector<double> > &strain, FFTW &fft){

    MMSP::grid<dim, fftw_complex> &ft_shape(shape);
    fft.execute(shape, ft_shape, FFTW_FORWARD);

    MMSP::vector<double> k_vec(3);
    mat3x3 eff_stress;
    khac.effective_stress(eff_stress);

    for (unsigned int i=0;i<3;i++){
        k_vec[i] = 0.0;
    }

    for (unsigned int node=0;node<MMSP::nodes(shape);node++){
        MMSP::vector<int> pos = shape.position(node);
        k_vector(pos, k_vec, dim);
        double k = norm(k_vec);

        if (abs(k) < 1E-6){
            continue;
        }

        divide(k_vec, k);

        fftw_complex force[3];
        generalized_force(eff_stress, k_vec, ft_shape(node), force);
    }
}