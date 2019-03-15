
cdef extern from "regression_kernels.hpp":
    cdef cppclass RegressionKernel:
    
        double evaluate(double x)

        double deriv(double x)