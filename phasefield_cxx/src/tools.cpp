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