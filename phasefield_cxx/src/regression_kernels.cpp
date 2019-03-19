#include "regression_kernels.hpp"
#include <cmath>

const double PI = acos(-1.0);

QuadraticKernel::QuadraticKernel(double width): RegressionKernel(), width(width){
    lower_limit = -width;
    upper_limit = width;
}
double QuadraticKernel::evaluate(double x) const{
    if (is_outside_support(x)){
        return 0.0;
    }

    return amplitude()*(1.0 - x*x/(width*width));
}

bool QuadraticKernel::is_outside_support(double x) const{
    return x < -width || x > width;
}

double QuadraticKernel::deriv(double x) const{
    if (is_outside_support(x)){
        return 0.0;
    }

    return -2*amplitude()*x/(width*width);
}

/** Gaussian kernel */

GaussianKernel::GaussianKernel(double std_dev): RegressionKernel(), std_dev(std_dev){
    lower_limit = -5*std_dev;
    upper_limit = 5*std_dev;
};

double GaussianKernel::evaluate(double x) const{
    double prefactor = 1.0/sqrt(2.0*PI*std_dev*std_dev);
    return prefactor*exp(-0.5*pow(x/std_dev, 2));
}

double GaussianKernel::deriv(double x) const{
    return x*evaluate(x)/pow(std_dev, 2);
}

bool GaussianKernel::is_outside_support(double x) const{
    return x < lower_limit || x > upper_limit;
}