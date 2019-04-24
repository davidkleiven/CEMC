#include "elasticity_util.hpp"
#include "tools.hpp"

void generalized_force(const mat3x3 &eff_stress, const double reciprocal_dir[3], fftw_complex shape_func, fftw_complex force[3]){
    for (unsigned int i=0;i<3;i++){
        real(force[i]) = 0.0;
        imag(force[i]) = 0.0;
        for (unsigned int j=0;j<3;j++){
            real(force[i]) += eff_stress[i][j]*reciprocal_dir[j]*real(shape_func);
            imag(force[i]) += eff_stress[i][j]*reciprocal_dir[j]*imag(shape_func);
        }
    }
}

unsigned int voigt_index(unsigned int i, unsigned int j){
    if (i == j){
        return i;
    }

    unsigned int min_index = i < j ? i : j;
    unsigned int max_index = (i == min_index) ? j : i;
    if ((min_index == 0) && (max_index == 1)){
        return 3;
    }
    else if ((min_index == 0) && (max_index == 2)){
        return 4;
    }
    else if ((min_index == 1) && (max_index == 2)){
        return 5;
    }

    stringstream ss;
    ss << "Cannot convert (" << i << ", " << j << ") to voigt index!";
    throw invalid_argument(ss.str());
}