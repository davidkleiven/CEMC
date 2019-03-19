# distutils: language = c++

from cemc.phasefield.cython.quadratic_kernel cimport QuadraticKernel
from cemc.phasefield.cython.regression_kernel cimport RegressionKernel
from cemc.phasefield.cython.gaussian_kernel cimport GaussianKernel

cdef class PyRegressionKernel:
    cdef RegressionKernel *thisptr

    def evaluate(self, x):
        raise NotImplementedError("Has to be implemented in derived classes!")

    def deriv(self, x):
        raise NotImplementedError("Has to be implemented in derived classes!")
        
cdef class PyQuadraticKernel(PyRegressionKernel):

    def __cinit__(self, double width):
        self.thisptr = new QuadraticKernel(width)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)

    def to_dict(self):
        return self.thisptr.to_dict()

    def from_dict(self, dict_repr):
        self.thisptr.from_dict(dict_repr)

cdef class PyGaussianKernel(PyRegressionKernel):

    def __cinit__(self, double std_dev):
        self.thisptr = new GaussianKernel(std_dev)

    def __dealloc__(self):
        del self.thisptr

    def evaluate(self, x):
        return self.thisptr.evaluate(x)

    def deriv(self, x):
        return self.thisptr.deriv(x)

    def to_dict(self):
        return self.thisptr.to_dict()

    def from_dict(self, dict_repr):
        self.thisptr.from_dict(dict_repr)