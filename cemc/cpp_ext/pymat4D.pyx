# distutils: language = c++

from cemc.cpp_ext.mat4D cimport Mat4D
cdef class PyMat4D:
    cdef Mat4D *matrix

    def __cinit__(self):
        self.matrix = new Mat4D()

    def __dealloc__(self):
        del self.matrix

    def from_numpy(self, npy):
        self.matrix.from_numpy(npy)

    def to_numpy(self):
        return self.matrix.to_numpy()