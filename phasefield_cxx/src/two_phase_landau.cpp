#include "two_phase_landau.hpp"
#include <cmath>

using namespace std;
TwoPhaseLandau::TwoPhaseLandau(){};

double TwoPhaseLandau::evaluate(double conc, const vector<double> &shape) const{
    double value = regressor->evaluate(conc);

    double x[4];
    x[0] = conc;
    memcpy(x+1, &shape[0], sizeof(double)*shape.size());
    value += polynomial->evaluate(x);
    return value;
}

double TwoPhaseLandau::partial_deriv_conc(double conc, const vector<double> &shape) const{
    double value = regressor->deriv(conc);
    double x[4];
    x[0] = conc;
    memcpy(x+1, &shape[0], sizeof(double)*shape.size());
    value += polynomial->deriv(x, 0);
    return value;
}

double TwoPhaseLandau::partial_deriv_shape(double conc, const std::vector<double> &shape, unsigned int direction) const{
    double x[4];
    x[0] = conc;
    memcpy(x+1, &shape[0], sizeof(double)*shape.size());
    return polynomial->deriv(x, direction+1);
}