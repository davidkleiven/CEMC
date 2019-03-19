#include "kernel_regressor.hpp"
#include<iostream>

using namespace std;

const double INF = 1E10;
int KernelRegressor::lower_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }
    double support = kernel->upper() - kernel->lower();
    
    return (x - xmin - support)/kernel_separation();
}

int KernelRegressor::upper_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }

    double support = kernel->upper() - kernel->lower();
    return (x - xmin + support)/kernel_separation();
}

double KernelRegressor::evaluate(double x) const{
    double value = 0.0;
    for (int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        if ((i >= coeff.size()) || (i < 0)) continue;

        value += coeff[i]*evaluate_kernel(i, x);
    }
    return value;
}

bool KernelRegressor::outside_domain(double x) const{
    return x < xmin || x > xmax;
}

double KernelRegressor::kernel_center(unsigned int i) const{
    return i*kernel_separation() + xmin;
}

double KernelRegressor::deriv(double x) const{
    double value = 0.0;
    for (int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        if ((i >= coeff.size()) || (i < 0)) continue;
        value += coeff[i]*kernel->deriv(x-kernel_center(i));
    }
    return value;
}

double KernelRegressor::kernel_separation() const{
    if (coeff.size() <= 1){
        return INF;
    }

    double domain_size = xmax - xmin;
    double dx = domain_size/(coeff.size() - 1);
    return dx;
}

double KernelRegressor::evaluate_kernel(unsigned int i, double x) const{
    return kernel->evaluate(x-kernel_center(i));
}