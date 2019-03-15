# distutils: language=c++
from cemc.phasefield.cython.quadratic_kernel cimport RegressionKernel
from libcpp.vector cimport vector

cdef extern from "kernel_regressor.hpp":
    cdef cppclass KernelRegressor:
        KernelRegressor(double xmin, double xmax)

        double evaluate(double x)

        double deriv(double x)

        void set_kernel(const RegressionKernel &kernel)

        void set_coeff(vector[double] coeff)

        double evaluate_kernel(unsigned int i, double x)

