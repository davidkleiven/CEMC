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
    return res;
}

double CahnHilliard::deriv(double x) const{
    double res = 0.0;
    unsigned int N = coeff.size();
    for (unsigned int i=0;i<N-1;i++){
        res += (N-i-1)*pow(x, N-i-2)*coeff[i];
    }
    return res;
}