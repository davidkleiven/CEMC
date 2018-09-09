# distutils: language = c++

from cemc.cpp_ext.pair_constraint cimport PairConstraint
from cython.operator cimport dereference as deref

cdef class PyPairConstraint:
    cdef PairConstraint *thisptr

    def __cinit__(self, PyCEUpdater upd, string cluster_name, string elm1,
                  string elm2):
        self.thisptr = new PairConstraint(deref(upd._cpp_class), cluster_name,
                                          elm1, elm2)

    def __dealloc__(self):
        del self.thisptr

    def elems_in_pair(self, system_changes):
        return self.thisptr.elems_in_pair(system_changes)
