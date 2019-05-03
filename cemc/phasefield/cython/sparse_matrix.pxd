from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "sparse_matrix.hpp":
    cdef cppclass SparseMatrix:
        void clear()

        void insert(unsigned int row, unsigned int col, double value)

        void dot(vector[double] &invec, vector[double] &out)

        void save(string &fname)

        bool is_symmetric()

        void to_csr()