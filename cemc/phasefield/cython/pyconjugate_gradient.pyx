from cemc.phasefield.cython.conjugate_gradient cimport ConjugateGradient
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef class PyConjugateGradient:
    cdef ConjugateGradient *thisptr

    def __cinit__(self, tol):
        self.thisptr = new ConjugateGradient(tol)

    def __dealloc__(self):
        del self.thisptr

    def solve(self, PySparseMatrix sp_mat, rhs, res):
        cdef vector[double] rhs_vec
        cdef vector[double] res_vec

        for i in range(len(rhs)):
            rhs_vec.push_back(rhs[i])
            res_vec.push_back(0.0)

        self.thisptr.solve(deref(sp_mat.thisptr), rhs_vec, res_vec)

        # Transfer back
        for i in range(len(res)):
            res[i] = res_vec[i]
        return res

