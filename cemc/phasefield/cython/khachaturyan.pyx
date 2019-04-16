# distutils: language = c++

from cemc.phasefield.cython.khachaturyan cimport Khachaturyan
cimport numpy as np
import numpy as np

cdef class PyKhachaturyan:
    cdef Khachaturyan *thisptr

    def __cinit__(self, int dim, elastic, misfit):
        self.thisptr = new Khachaturyan(dim, elastic, misfit)


    def __dealloc__(self):
        del self.thisptr

    def green_function(self, direction):
        return self.thisptr.green_function(direction)

    def wave_vector(self, indx, N):
        cdef unsigned int indx_c[3]
        cdef double dir_c[3]

        for i in range(3):
            indx_c[i] = indx[i]
        
        self.thisptr.wave_vector(indx_c, dir_c, N)
        out = np.zeros(3)
        for i in range(3):
            out[i] = dir_c[i]
        return out

    def zeroth_order_integral(self, ft_shp):
        return self.thisptr.zeroth_order_integral(ft_shp)

    def effective_stress(self):
        cdef double e_stress[3][3]
        self.thisptr.effective_stress(e_stress)

        stress = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                stress[i, j] = e_stress[i][j]
        return stress
        