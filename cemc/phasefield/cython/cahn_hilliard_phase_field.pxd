# distutils: language = c++

cdef extern from "cahn_hilliard.hpp":
    cdef cppclass CahnHilliard

from libcpp.string cimport string
cdef extern from "cahn_hilliard_phase_field.hpp":
    cdef cppclass CahnHilliardPhaseField[T]:
        CahnHilliardPhaseField(int L, string prefix, CahnHilliard *free_eng, \
                                 double M, double dt, double alpha)

        void update(unsigned int steps)

        void run(unsigned int nsteps, int increment)

        void random_initialization(double lower, double upper)