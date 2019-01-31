# distutils: language=c++

from libcpp.vector cimport vector

cdef extern from "two_phase_landau.hpp":
  cdef cppclass TwoPhaseLandau:
      TwoPhaseLandau(double c1, double c2, vector[double] &coeff)

      double evaluate(double conc, vector[double] &shape)