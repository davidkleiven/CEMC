# distutils: language = c++

from cemc.phasefield.cython.quadratic_kernel cimport QuadraticKernel

cdef class PyQuadraticKernel:
    cdef QuadraticKernel *thisptr

    def __cinit__(self, double width):
        self.thisptr = new QuadraticKernel(width)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)