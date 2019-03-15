# distutils: language=c++

cdef extern from "regression_kernels.hpp":
    cdef cppclass QuadraticKernel:
        QuadraticKernel(double width)

        double evaluate(double x)

        double deriv(double x)