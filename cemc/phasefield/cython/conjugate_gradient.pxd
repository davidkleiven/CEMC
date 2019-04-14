from libcpp.vector cimport vector

cdef extern from "sparse_matrix.hpp":
    cdef cppclass SparseMatrix:
        pass

cdef extern from "conjugate_gradient.hpp":
    cdef cppclass ConjugateGradient:
        ConjugateGradient(double tol)
        void solve(SparseMatrix &mat, vector[double] &rhs, vector[double] &out) except+