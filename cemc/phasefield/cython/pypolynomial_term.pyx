# distutils: language = c++

from cemc.phasefield.cython.polynomial_term cimport PolynomialTerm
from libcpp.vector cimport vector


cdef class PyPolynomialTerm:
    cdef PolynomialTerm *thisptr

    def __cinit__(self, inner_power, outer_power):
        cdef vector[unsigned int] vec
        for x in inner_power:
            vec.push_back(x)
        self.thisptr = new PolynomialTerm(vec, outer_power)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        if len(x) > 10:
            raise ValueError("We only support up to 10 dimensional terms")
        cdef double c_array[10]
        for i in range(len(x)):
            c_array[i] = x[i]
        return self.thisptr.evaluate(c_array)

    def deriv(self, x, crd):
        if len(x) > 10:
            raise ValueError("We only support up to 10 dimensional terms")
        cdef double c_array[10]
        for i in range(len(x)):
            c_array[i] = x[i]
        return self.thisptr.deriv(c_array, crd)