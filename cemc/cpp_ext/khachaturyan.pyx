# distutils: language = c++

from cemc.cpp_ext.khachaturyan cimport Khachaturyan
cimport numpy as np
import numpy as np

cdef class PyKhachaturyan:
    cdef Khachaturyan *_self

    def __cinit__(self, ft_shp, elastic, misfit):
        self._self = new Khachaturyan(ft_shp, elastic, misfit)

    def __dealloc__(self):
        del self._self

    def green_function(self, direction):
        return self._self.green_function(direction)

    def wave_vector(self, indx):
        cdef unsigned int indx_c[3]
        cdef double dir_c[3]

        for i in range(3):
            indx_c[i] = indx[i]
        
        self._self.wave_vector(indx_c, dir_c)
        out = np.zeros(3)
        for i in range(3):
            out[i] = dir_c[i]
        return out

    def zeroth_order_integral(self):
        return self._self.zeroth_order_integral()
        