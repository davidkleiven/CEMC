#include "tools.hpp"
#include <cmath>

const double PI = acos(-1.0);

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

double norm(const MMSP::vector<double> &vec){
    double value = 0.0;
    for (unsigned int i=0;i<vec.length();i++){
        value += pow(vec[i], 2);
    }
    return sqrt(value);
}