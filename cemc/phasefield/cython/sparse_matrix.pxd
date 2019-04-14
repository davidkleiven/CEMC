from libcpp.vector cimport vector

cdef extern from "sparse_matrix.hpp":
    cdef cppclass SparseMatrix:
        void clear()

        void insert(unsigned int row, unsigned int col, double value)

        void dot(vector[double] &invec, vector[double] &out)