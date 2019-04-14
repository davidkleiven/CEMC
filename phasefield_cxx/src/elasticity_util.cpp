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