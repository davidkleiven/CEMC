#include "elasticity_util.hpp"

void generalized_force(const mat3x3 &eff_stress, const double reciprocal_dir[3], const fftw_complex &shape_func, fftw_complex force[3]){
    for (unsigned int i=0;i<3;i++){
        force[i].re = 0.0;
        force[i].im = 0.0;
        for (unsigned int j=0;j<3;j++){
            force[i].re += eff_stress[i][j]*reciprocal_dir[j]*shape_func.re;
            force[i].im += eff_stress[i][j]*reciprocal_dir[j]*shape_func.im;
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