# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii

cdef extern from "init_numpy.hpp":
    pass
    
cimport numpy as np
#np.import_array()

include "pymat4D.pyx"
include "khachaturyan.pyx"
include "pytwo_phase_landau.pyx"