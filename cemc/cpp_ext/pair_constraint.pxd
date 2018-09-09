# distutils: language = c++

from ce_updater cimport CEUpdater
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "pair_constraint.hpp":
    cdef cppclass PairConstraint:
        PairConstraint(CEUpdater &updater, string &cluster_name,
                       string &elemt1, string &elem2)

        bool elems_in_pair(object system_changes)
