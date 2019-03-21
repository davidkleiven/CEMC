# distutils: language = c++

from cemc.phasefield.cython.two_phase_landau cimport TwoPhaseLandau
from libcpp.vector cimport vector
from cython.operator cimport dereference

cdef class PyTwoPhaseLandau:
    cdef TwoPhaseLandau *thisptr

    def __cinit__(self):
        self.thisptr = new TwoPhaseLandau()

    def evaluate(self, conc, shape):
        cdef vector[double] shp_vec
        for s in shape:
            shp_vec.push_back(s)
        return self.thisptr.evaluate(conc, shp_vec)

    def set_kernel_regressor(self, PyKernelRegressor pyregr):
        self.thisptr.set_kernel_regressor(dereference(pyregr.thisptr))

    def set_polynomial(self, PyPolynomial pypoly):
        self.thisptr.set_polynomial(dereference(pypoly.thisptr))

    def partial_deriv_conc(self, conc, shape):
        cdef vector[double] shp_vec
        for i in range(len(shape)):
            shp_vec.push_back(shape[i])
        return self.thisptr.partial_deriv_conc(conc, shp_vec)

    def partial_deriv_shape(self, conc, shape, direction):
        cdef vector[double] shp_vec
        for i in range(len(shape)):
            shp_vec.push_back(shape[i])

        if direction < 0 or direction >= 3:
            raise ValueError("The direction has to be between 0 and 3!")
        return self.thisptr.partial_deriv_shape(conc, shp_vec, direction)
