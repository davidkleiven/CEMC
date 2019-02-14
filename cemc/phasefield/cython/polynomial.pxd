# distutils: language = c++

cdef extern from "polynomial_term.hpp":
    cdef cppclass PolynomialTerm:
        pass

cdef extern from "polynomial.hpp":
    cdef cppclass Polynomial:
        Polynomial(unsigned int dim)

        double add_term(double coeff, PolynomialTerm &term)

        double evaluate(double [])

        double deriv(double [], unsigned int crd)