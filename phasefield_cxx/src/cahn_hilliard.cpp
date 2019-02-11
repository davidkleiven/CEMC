#include "cahn_hilliard.hpp"
#include <cmath>

using namespace std;

CahnHilliard::CahnHilliard(const vector<double> &coefficients): coeff(coefficients){};

double CahnHilliard::evaluate(double x) const{
    double res = 0.0;
    unsigned int N = coeff.size();
    for (unsigned int i=0;i<coeff.size();i++){
        res += pow(x, N-i-1)*coeff[i];
    }

    if (has_bounds){
        res += penalty*regularization(x);
    }
    return res;
}

double CahnHilliard::deriv(double x) const{
    double res = 0.0;
    unsigned int N = coeff.size();
    for (unsigned int i=0;i<N-1;i++){
        res += (N-i-1)*pow(x, N-i-2)*coeff[i];
    }

    if (has_bounds){
        res += penalty*regularization_deriv(x);
    }
    return res;
}

void CahnHilliard::set_bounds(double lower, double upper){
    has_bounds = true;
    lower_bound = lower;
    upper_bound = upper;
}

bool CahnHilliard::is_outside_range(double x) const{
    return x >= upper_bound || x < lower_bound;
}

double CahnHilliard::regularization(double x) const{
    if (!is_outside_range(x)){
        return 0.0;
    }

    double range = upper_bound - lower_bound;
    double scale = rng_scale*range;

    if (x >= upper_bound){
        return exp((x-upper_bound)/scale);
    }
    else if (x < lower_bound){
        return exp((lower_bound-x)/scale);
    }
    return 0.0;
}

double CahnHilliard::regularization_deriv(double x) const{
    if (!is_outside_range(x)){
        return 0.0;
    }

    double range = upper_bound - lower_bound;
    double scale = rng_scale*range;
    if (x > upper_bound){
        return exp((x-upper_bound)/scale)/scale;
    }
    else if(x < lower_bound){
        return -exp((lower_bound-x)/scale)/scale;
    }
    return 0.0;
}