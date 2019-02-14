#include "polynomial_term.hpp"
#include <cmath>

PolynomialTerm::PolynomialTerm(const uintvec_t &i_power, unsigned int outer_power): outer_power(outer_power){
        dim = i_power.size();

        inner_power = new unsigned int[dim];
        centers = new double[dim];

        for (unsigned int i=0;i<dim;i++){
            inner_power[i] = i_power[i];
            centers[i] = 0.0;
        }
    };

PolynomialTerm::PolynomialTerm(unsigned int dim, unsigned int i_power, unsigned int outer_power): dim(dim), outer_power(outer_power){
    inner_power = new unsigned int[dim];
    centers = new double[dim];

    for (unsigned int i=0;i<dim;i++){
        inner_power[i] = i_power;
        centers[i] = 0.0;
    }
};

PolynomialTerm::~PolynomialTerm(){
    delete [] inner_power;
    delete [] centers;
}

double PolynomialTerm::evaluate(double x[]) const{
    return pow(evaluate_inner(x), outer_power);
}

double PolynomialTerm::evaluate_inner(double x[]) const{
    double value = 0.0;
    for (unsigned int i=0;i<dim;i++){
        value += pow(x[i] - centers[i], inner_power[i]);
    }
    return value;
}

double PolynomialTerm::deriv(double x[], unsigned int crd) const{
    if ((outer_power == 0) || (inner_power[crd] == 0)){
        return 0.0;
    }

    return outer_power*pow(evaluate_inner(x), outer_power-1)*inner_power[crd]*pow(x[crd] - centers[crd], inner_power[crd]-1);
}