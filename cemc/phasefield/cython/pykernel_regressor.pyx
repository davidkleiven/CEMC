# distutils: language = c++

from cemc.phasefield.cython.kernel_regressor cimport KernelRegressor
from cython.operator cimport dereference
from libcpp.vector cimport vector

cdef class PyKernelRegressor:
    cdef KernelRegressor *thisptr

    def __cinit__(self, double xmin, double xmax):
        self.thisptr = new KernelRegressor(xmin, xmax)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)

    def set_kernel(self, PyRegressionKernel kernel):
        self.thisptr.set_kernel(dereference(kernel.thisptr))

    def set_coeff(self, coeff):
        cdef vector[double] vec_coeff
        for i in range(len(coeff)):
            vec_coeff.push_back(coeff[i])
        self.thisptr.set_coeff(vec_coeff)