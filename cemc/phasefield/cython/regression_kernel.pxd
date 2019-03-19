
cdef extern from "regression_kernels.hpp":
    cdef cppclass RegressionKernel:
    
        double evaluate(double x)

        double deriv(double x)

        object to_dict()

        void from_dict(object dict_repr) except+