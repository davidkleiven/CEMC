# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii

cdef extern from "init_numpy.hpp":
    pass

cimport numpy as np
np.import_array()

include "pymat4D.pyx"
include "khachaturyan.pyx"
include "pytwo_phase_landau.pyx"
include "pycahn_hilliard.pyx"
include "pycahn_hilliard_phase_field.pyx"

# Hack: MMSP contains implementation in the header files
# therefore we need to include the cpp file here
cdef extern from "mmsp_files.cpp":
    pass