# distutils: language=c++

from cemc.phasefield.cython.regression_kernel cimport RegressionKernel

cdef extern from "regression_kernels.hpp":
    cdef cppclass QuadraticKernel(RegressionKernel):
        QuadraticKernel(double width)

        double evaluate(double x)

        double deriv(double x)