# distutils: language = c++

from libcpp.string cimport string
from cemc.cpp_ext.ce_updater cimport CEUpdater

cdef extern from "cluster_tracker.hpp":
  cdef cppclass ClusterTracker:
      ClusterTracker(CEUpdater &updater, string &cname, string &element)

      void find_clusters()

      object get_cluster_statistics_python()

      object atomic_clusters2group_indx_python()

      void grow_cluster(unsigned int size)

      object surface_python()
