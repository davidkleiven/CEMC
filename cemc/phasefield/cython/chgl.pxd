# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "chgl.hpp":
    cdef cppclass CHGL[T]:
        CHGL(int L, string &prefix, unsigned int num_gl_fields, \
             double M, double alpha, double dt, double gl_damping, 
             const vector[vector[double]] &interface) except +

        void update(int steps)

        void run(unsigned int start, unsigned int nsteps, int increment)

        void random_initialization(unsigned int field, double lower, double upper)

        void from_file(string fname)

        void from_npy_array(object fields) except +

        object to_npy_array() except +