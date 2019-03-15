#ifndef KERNEL_REGRESSOR_H
#define KERNEL_REGRESSOR_H

#include "regression_kernels.hpp"
#include <vector>

class KernelRegressor{
    public:
        KernelRegressor(double xmin, double xmax): xmin(xmin), xmax(xmax){};

        void set_kernel(const RegressionKernel &new_kernel){kernel = &new_kernel;};

        /** Evaluate */
        double evaluate(double x) const;

        /** Calculate the derivative */
        double deriv(double x) const;

        /** Set the coefficients */
        void set_coeff(const std::vector<double> &new_coeff){coeff = new_coeff;};
    private:
        const RegressionKernel *kernel{nullptr};
        std::vector<double> coeff;
        double xmin{0.0};
        double xmax{0.0};

        unsigned int lower_non_zero_kernel(double x) const;
        unsigned int upper_non_zero_kernel(double x) const;
        bool outside_domain(double x) const;
        double kernel_center(unsigned int indx) const;

        double kernel_separation() const;
};
#endif