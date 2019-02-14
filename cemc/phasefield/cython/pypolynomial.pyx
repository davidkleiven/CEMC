# distutils: language = c++

from cemc.phasefield.cython.polynomial cimport Polynomial
from cython.operator cimport dereference

cdef class PyPolynomial:
    cdef Polynomial *thisptr

    def __cinit__(self, dim):
        self.thisptr = new Polynomial(dim)

    def __dealloc__(self):
        del self.thisptr

    def add_term(self, coeff, PyPolynomialTerm new_term):
        self.thisptr.add_term(coeff, dereference(new_term.thisptr))

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
