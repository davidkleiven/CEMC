# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from cemc.cpp_ext.ce_updater cimport CEUpdater

cdef extern from "cluster_tracker.hpp":
  cdef cppclass ClusterTracker:
      ClusterTracker(CEUpdater &updater, vector[string] &cname, vector[string] &element) except +

      void find_clusters()

      object get_cluster_statistics_python() except+

      object atomic_clusters2group_indx_python()

      object surface_python()

      bool move_creates_new_cluster(object system_changes)

      void update_clusters(object system_changes) except+
