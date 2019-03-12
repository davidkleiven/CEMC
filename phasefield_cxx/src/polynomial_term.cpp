#include "polynomial_term.hpp"
#include <cmath>

PolynomialTerm::PolynomialTerm(const uintvec_t &i_power){
        dim = i_power.size();

        inner_power = new unsigned int[dim];
        centers = new double[dim];

        for (unsigned int i=0;i<dim;i++){
            inner_power[i] = i_power[i];
            centers[i] = 0.0;
        }
    };

PolynomialTerm::PolynomialTerm(unsigned int dim, unsigned int i_power): dim(dim){
    inner_power = new unsigned int[dim];
    centers = new double[dim];

    for (unsigned int i=0;i<dim;i++){
        inner_power[i] = i_power;
        centers[i] = 0.0;
    }
};

PolynomialTerm::PolynomialTerm(const PolynomialTerm &other){
    this->swap(other);
}

PolynomialTerm& PolynomialTerm::operator=(const PolynomialTerm &other){
    this->swap(other);
    return *this;
}


void PolynomialTerm::swap(const PolynomialTerm &other){
    delete [] inner_power;
    delete [] centers;

    dim = other.dim;

    inner_power = new unsigned int[dim];
    centers = new double[dim];

    for (unsigned int i=0;i<dim;i++){
        inner_power[i] = other.inner_power[i];
        centers[i] = other.centers[i];
    }
}

PolynomialTerm::~PolynomialTerm(){
    delete [] inner_power;
    delete [] centers;
}

double PolynomialTerm::evaluate(double x[]) const{
    double value = 1.0;
    for (unsigned int i=0;i<dim;i++){
        value *= pow(x[i] - centers[i], inner_power[i]);
    }
    return value;
}

double PolynomialTerm::deriv(double x[], unsigned int crd) const{
    if (inner_power[crd] == 0){
        return 0.0;
    }
    double value = inner_power[crd]*pow(x[crd] - centers[crd], inner_power[crd]-1);
    for (unsigned int i=0;i<dim;i++){
        if (i != crd){
            value *= pow(x[i] - centers[i], inner_power[i]);
        }
    }
    return value;
}