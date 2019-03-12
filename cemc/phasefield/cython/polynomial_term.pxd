# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "polynomial_term.hpp":
    cdef cppclass PolynomialTerm:
        PolynomialTerm(vector[unsigned int] inner_power)
        PolynomialTerm(unsigned int dim, unsigned int inner_power)

        double evaluate(double x[])
        double deriv(double x[], unsigned int crd)