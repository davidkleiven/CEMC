#include "tools.hpp"
#include <cmath>
#include <omp.h>

const double PI = acos(-1.0);

using namespace std;

void k_vector(const MMSP::vector<int> &pos, MMSP::vector<double> &k_vec, int N){
    for (unsigned int i=0;i<pos.length();i++){
        if (pos[i] < N/2){
            k_vec[i] = PI*pos[i]/N;
        }
        else{
            k_vec[i] = -PI*(N-pos[i])/N;
        }
    }
}

void dot(const mat3x3 &mat1, const MMSP::vector<double> &vec, MMSP::vector<double> &out){
    for (unsigned int i=0;i<3;i++){
        out[i] = 0.0;
    }

    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        out[i] += mat1[i][j]*vec[j];
    }
}

double dot(const MMSP::vector<double> &vec1, const MMSP::vector<double> &vec2){
    double value = 0.0;
    for (unsigned int i=0;i<vec1.length();i++){
        value += vec1[i]*vec2[i];
    }
    return value;
}

double dot(const vector<double> &v1, const vector<double> &v2){
    double inner_prod = 0.0;
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(+ : inner_prod)
    #endif
    for (unsigned int i=0;i<v1.size();i++){
        inner_prod += v1[i]*v2[i];
    }
    return inner_prod;
}

double norm(const MMSP::vector<double> &vec){
    double value = 0.0;
    for (unsigned int i=0;i<vec.length();i++){
        value += pow(vec[i], 2);
    }
    return sqrt(value);
}

void inplace_minus(vector<double> &vec1, const vector<double> &vec2){
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<vec1.size();i++){
        vec1[i] -= vec2[i];
    }
}

double inf_norm(const vector<double> &vec){
    double max_val = abs(vec[0]);
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(max : max_val)
    #endif
    for (unsigned int i=0;i<vec.size();i++){
        max_val = abs(vec[i]) > max_val ? abs(vec[i]) : max_val;
    }
    return max_val;
}

double least_squares_slope(double x[], double y[], unsigned int N){
    double Sxx = 0.0;
    double Sxy = 0.0;
    double Sx = 0.0;
    double Sy = 0.0;

    for (unsigned int i=0;i<N;i++){
        Sx += x[i];
        Sy += y[i];
        Sxy += x[i]*y[i];
        Sxx += x[i]*x[i];
    }

    return (N*Sxy - Sx*Sy)/(N*Sxx - Sx*Sx);
}

double contract_tensors(const mat3x3 &mat1, const mat3x3 &mat2){
    double value = 0.0;
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        value += mat1[i][j]*mat2[i][j];
    }
    return value;
}

bool is_origin(MMSP::vector<double> &dir){
    double tol = 1E-6;
    bool origin = true;
    for (int i=0;i<dir.length();i++){
        origin = abs(dir[i]) < tol;
    }
    return origin;
}

double B_tensor_element(MMSP::vector<double> &dir, const mat3x3 &green, \
                        const mat3x3 &eff_stress1, const mat3x3 &eff_stress2)
{
    MMSP::vector<double> temp_vec(3);
    dot(eff_stress2, dir, temp_vec);

    MMSP::vector<double> temp_vec2(3);
    dot(green, temp_vec, temp_vec2);
    dot(eff_stress1, temp_vec2, temp_vec);
    return dot(dir, temp_vec);
}

double B_tensor_element_origin(const mat3x3 &green, const mat3x3 &eff_stress1, const mat3x3 &eff_stress2, \
                               std::vector< MMSP::vector<double> > &directions)
{
    double value = 0.0;
    for (auto& dir : directions){
        value += B_tensor_element(dir, green, eff_stress1, eff_stress2);
    }
    return value/directions.size();
}