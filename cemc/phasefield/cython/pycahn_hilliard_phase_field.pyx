# distutils: language = c++

from cemc.phasefield.cython cimport cahn_hilliard_phase_field as chpf
from cemc.phasefield.cython.cahn_hilliard cimport CahnHilliard
from libcpp.string cimport string

# Hack to support integer template arguments with cython
cdef extern from *:
    ctypedef int intParameter1 "1"
    ctypedef int intParameter2 "2"
    ctypedef int intParameter3 "3"

cdef class PyCahnHilliardPhaseField:
    cdef chpf.CahnHilliardPhaseField[intParameter1] *thisptr1D
    cdef chpf.CahnHilliardPhaseField[intParameter2] *thisptr2D
    cdef chpf.CahnHilliardPhaseField[intParameter3] *thisptr3D
    cdef int dim

    def __cinit__(self, dim, L, prefix, PyCahnHilliard free_eng, M, dt, alpha):
        if dim == 1:
            self.thisptr1D = new chpf.CahnHilliardPhaseField[intParameter1](L, prefix, free_eng.thisptr, M, dt, alpha)
        elif dim == 2:
            self.thisptr2D = new chpf.CahnHilliardPhaseField[intParameter2](L, prefix, free_eng.thisptr, M, dt, alpha)
        elif dim == 3:
            self.thisptr3D = new chpf.CahnHilliardPhaseField[intParameter3](L, prefix, free_eng.thisptr, M, dt, alpha)
        else:
            raise ValueError("dim has to be one 1, 2 or 3")

    def __init__(self, dim, L, prefix, free_eng, M, dt, alpha):
        self.dim = dim

    def __dealloc__(self):
        del self.thisptr1D
        del self.thisptr2D
        del self.thisptr3D

    def run(self, nsteps, increment, start=0):
        if self.dim == 1:
            self.thisptr1D.run(start, nsteps, increment)
        elif self.dim == 2:
            self.thisptr2D.run(start, nsteps, increment)
        elif self.dim == 3:
            self.thisptr3D.run(start, nsteps, increment)

    def random_initialization(self, lower, upper):
        if self.dim == 1:
            self.thisptr1D.random_initialization(lower, upper)
        elif self.dim == 2:
            self.thisptr2D.random_initialization(lower, upper)
        elif self.dim == 3:
            self.thisptr3D.random_initialization(lower, upper)
        else:
            raise ValueError("Dimension has to be 1, 2, or 3")

    def from_file(self, fname):
        if self.dim == 1:
            self.thisptr1D.from_file(fname)
        elif self.dim == 2:
            self.thisptr2D.from_file(fname)
        elif self.dim == 3:
            self.thisptr3D.from_file(fname)
        else:
            raise ValueError("Dimension has to be 1, 2, or 3")

    def from_npy_array(self, npy_array):
        if self.dim == 1:
            self.thisptr1D.from_npy_array([npy_array])
        elif self.dim == 2:
            self.thisptr2D.from_npy_array([npy_array])
        elif self.dim == 3:
            self.thisptr3D.from_npy_array([npy_array])

    def to_npy_array(self):
        if self.dim == 1:
            return self.thisptr1D.to_npy_array()[0]
        elif self.dim == 2:
            return self.thisptr2D.to_npy_array()[0]
        elif self.dim == 3:
            return self.thisptr3D.to_npy_array()[0]

    def set_adaptive(self, min_step, max_change):
        if self.dim == 1:
            return self.thisptr1D.set_adaptive(min_step, max_change)
        elif self.dim == 2:
            return self.thisptr2D.set_adaptive(min_step, max_change)
        elif self.dim == 3:
            return self.thisptr3D.set_adaptive(min_step, max_change)

    def build2D(self):
        if self.dim == 1:
            return self.thisptr1D.build2D()
        elif self.dim == 2:
            return self.thisptr2D.build2D()
        elif self.dim == 3:
            return self.thisptr3D.build2D()