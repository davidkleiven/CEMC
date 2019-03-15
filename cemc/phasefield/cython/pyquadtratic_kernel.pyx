# distutils: language = c++

from cemc.phasefield.cython.quadratic_kernel cimport QuadraticKernel

cdef class PyQuadtraticKernel:
    cdef QuadraticKernel *thisptr

    def __cinit__(double width):
        thisptr = new QuadraticKernel(width)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)