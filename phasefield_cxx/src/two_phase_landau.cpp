#include "two_phase_landau.hpp"
#include <cmath>

using namespace std;
TwoPhaseLandau::TwoPhaseLandau(double c1, double c2, const vector<double> &coefficients): \
    c1(c1), c2(c2), coeff(coefficients){};

double TwoPhaseLandau::evaluate(double conc, const vector<double> &shape) const{
    double sum_sq = 0.0;
    double sum_quad = 0.0;

    for (double s : shape){
        sum_sq += s*s;
        sum_quad += pow(s, 4);
    }
    return coeff[0]*pow(conc - c1, 2) + coeff[1]*(conc - c2)*sum_sq + coeff[2]*sum_quad + coeff[3]*pow(sum_sq, 3);
}