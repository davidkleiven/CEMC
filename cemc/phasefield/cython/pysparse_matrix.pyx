from cemc.phasefield.cython.sparse_matrix cimport SparseMatrix
from libcpp.vector cimport vector

cdef class PySparseMatrix:
    cdef SparseMatrix *thisptr

    def __cinit__(self):
        self.thisptr = new SparseMatrix()

    def __dealloc__(self):
        del self.thisptr

    def insert(self, row, col, value):
        self.thisptr.insert(row, col, value)

    def dot(self, vec):
        # Transfer the arrays to C++ vectors
        cdef vector[double] v1
        cdef vector[double] v2

        for i in range(len(vec)):
            v1.push_back(vec[i])
            v2.push_back(0.0)

        self.thisptr.dot(v1, v2)

        # Transfer back
        out = []
        for i in range(len(vec)):
            out.append(v2[i])
        return out

    def clear(self):
        self.thisptr.clear()

    def save(self, fname):
        self.thisptr.save(fname)

    def is_symmetric(self):
        return self.thisptr.is_symmetric()