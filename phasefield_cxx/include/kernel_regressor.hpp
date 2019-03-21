#ifndef KERNEL_REGRESSOR_H
#define KERNEL_REGRESSOR_H

#include "regression_kernels.hpp"
#include <vector>
#include <Python.h>

class KernelRegressor{
    public:
        KernelRegressor(double xmin, double xmax): xmin(xmin), xmax(xmax){};

        void set_kernel(const RegressionKernel &new_kernel){kernel = &new_kernel;};

        /** Return true if the kernel is set */
        bool kernel_is_set() const{ return kernel != nullptr; }

        /** Evaluate */
        double evaluate(double x) const;

        /** Calculate the derivative */
        double deriv(double x) const;

        /** Set the coefficients */
        void set_coeff(const std::vector<double> &new_coeff){coeff = new_coeff;};

        /** Get the value of a single kernel */
        double evaluate_kernel(unsigned int i, double x) const;

        /** Return a dictionary representation of the object */
        PyObject *to_dict() const;

        /** Initialize the object from a dictionary */
        void from_dict(PyObject *dict_repr);
    private:
        const RegressionKernel *kernel{nullptr};
        std::vector<double> coeff;
        double xmin{0.0};
        double xmax{0.0};

        int lower_non_zero_kernel(double x) const;
        int upper_non_zero_kernel(double x) const;
        bool outside_domain(double x) const;
        double kernel_center(unsigned int indx) const;

        double kernel_separation() const;
};
#endif