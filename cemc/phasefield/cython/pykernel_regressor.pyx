# distutils: language = c++

from cemc.phasefield.cython.kernel_regressor cimport KernelRegressor
from cython.operator cimport dereference
from libcpp.vector cimport vector
import numpy as np

cdef class PyKernelRegressor:
    cdef KernelRegressor *thisptr

    def __cinit__(self, double xmin, double xmax):
        self.thisptr = new KernelRegressor(xmin, xmax)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        if np.isscalar(x):
            return self.thisptr.evaluate(x)
        return [self.thisptr.evaluate(x[i]) for i in range(len(x))]

    def deriv(self, x):
        if np.isscalar(x):
            return self.thisptr.deriv(x)
        return [self.thisptr.deriv(x[i]) for i in range(len(x))]

    def set_kernel(self, PyRegressionKernel kernel):
        self.thisptr.set_kernel(dereference(kernel.thisptr))

    def evaluate_kernel(self, i, x):
        if np.isscalar(x):
            return self.thisptr.evaluate_kernel(i, x)    
        
        return [self.thisptr.evaluate_kernel(i, x[j]) for j in range(len(x))]

    def set_coeff(self, coeff):
        cdef vector[double] vec_coeff
        for i in range(len(coeff)):
            vec_coeff.push_back(coeff[i])
        self.thisptr.set_coeff(vec_coeff)