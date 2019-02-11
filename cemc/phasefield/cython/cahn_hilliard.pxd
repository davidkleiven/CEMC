# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "cahn_hilliard.hpp":
    cdef cppclass CahnHilliard:
        CahnHilliard(vector[double] coeff)

        double evaluate(double x)

        double deriv(double x)

        double regularization(double x)

        double regularization_deriv(double x)

        void set_bounds(double lower, double upper)

        void set_penalty(double penalty)

        void set_range_scale(double scale)