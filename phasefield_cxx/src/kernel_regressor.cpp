#include "kernel_regressor.hpp"

unsigned int KernelRegressor::lower_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }
    double support = kernel->upper() - kernel->lower();
    double domain_size = xmax - xmin;
    double dx = domain_size/(coeff.size() + 1);
    return (x - xmin - support)/dx;
}

unsigned int KernelRegressor::upper_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }

    double support = kernel->upper() - kernel->lower();
    double domain_size = xmax - xmin;
    double dx = domain_size/coeff.size();
    return (x - xmin + support)/dx;
}

double KernelRegressor::evaluate(double x) const{
    double value = 0.0;
    for (unsigned int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        value += coeff[i]*kernel->evaluate(x-kernel_center(i));
    }
    return value;
}

bool KernelRegressor::outside_domain(double x) const{
    return x < xmin || x > xmax;
}

double KernelRegressor::kernel_center(unsigned int i) const{
    double domain_size = xmax - xmin;
    double dx = domain_size/(coeff.size() + 1);
    return (i+1)*dx + xmin;
}

double KernelRegressor::deriv(double x) const{
    double value = 0.0;
    for (unsigned int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        value += coeff[i]*kernel->deriv(x-kernel_center(i));
    }
    return value;
}
