# distutils: language = c++

from cemc.phasefield.cython.cahn_hilliard cimport CahnHilliard
from libcpp.vector cimport vector

cdef class PyCahnHilliard:
    cdef CahnHilliard *thisptr

    def __cinit__(self, coeff):
        cdef vector[double] c_vec
        for c in coeff:
            c_vec.push_back(c)
        self.thisptr = new CahnHilliard(c_vec)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)