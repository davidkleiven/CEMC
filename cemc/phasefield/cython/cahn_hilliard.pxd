# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "cahn_hilliard.hpp":
    cdef cppclass CahnHilliard:
        CahnHilliard(vector[double] coeff)

        double evaluate(double x)

        double deriv(double x)