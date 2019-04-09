# distutils: language=c++

from libcpp.vector cimport vector
from cemc.phasefield.cython.kernel_regressor cimport KernelRegressor
from cemc.phasefield.cython.polynomial cimport Polynomial

cdef extern from "two_phase_landau.hpp":
  cdef cppclass TwoPhaseLandau:
      TwoPhaseLandau()

      double evaluate(double conc, vector[double] &shape)

      double partial_deriv_conc(double conc, vector[double] &shape)

      double partial_deriv_shape(double conc, vector[double] &shape, unsigned int direction)

      void set_kernel_regressor(KernelRegressor &regresssor)

      void set_polynomial(Polynomial &poly)

      void set_discontinuity(double conc, double jump)