#include "regression_kernels.hpp"


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
