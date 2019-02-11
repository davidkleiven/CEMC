# distutils: language = c++

from cemc.phasefield.cython.cahn_hilliard cimport CahnHilliard
from libcpp.vector cimport vector

cdef class PyCahnHilliard:
    cdef CahnHilliard *thisptr

    def __cinit__(self, coeff, penalty=100.0, bounds=None, range_scale=0.1):
        cdef vector[double] c_vec
        for c in coeff:
            c_vec.push_back(c)
        self.thisptr = new CahnHilliard(c_vec)

    def __init__(self, coeff, penalty=100.0, bounds=None, range_scale=0.1):
        self.thisptr.set_penalty(penalty)
        self.thisptr.set_range_scale(range_scale)

        if bounds is not None:
            self.thisptr.set_bounds(bounds[0], bounds[1])

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)

    def regularization(self, x):
        return self.thisptr.regularization(x)

    def regularization_deriv(self, x):
        return self.thisptr.regularization_deriv(x)

    def set_bounds(self, lower, upper):
        self.thisptr.set_bounds(lower, upper)

    def set_penalty(self, penalty):
        self.thisptr.set_penalty(penalty)

    def set_range_scale(self, scale):
        self.thisptr.set_range_scale(scale)