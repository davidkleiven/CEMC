# distutils: language = c++

from cemc.phasefield.cython.two_phase_landau cimport TwoPhaseLandau
from libcpp.vector cimport vector

cdef class PyTwoPhaseLandau:
    cdef TwoPhaseLandau *thisptr

    def __cinit__(self, c1, c2, coeff):
        cdef vector[double] coeff_vec
        for c in coeff:
            coeff_vec.push_back(c)
        self.thisptr = new TwoPhaseLandau(c1, c2, coeff_vec)

    def evaluate(self, conc, shape):
        cdef vector[double] shp_vec
        for s in shape:
            shp_vec.push_back(s)
        return self.thisptr.evaluate(conc, shp_vec)