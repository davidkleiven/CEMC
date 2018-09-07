# distutils: language = c++
# distutils: sources =  cpp/src/cluster_tracker.cpp

from cemc.cpp_ext.cluster_tracker cimport ClusterTracker
from cemc.cpp_ext.ce_updater cimport CEUpdater
from libcpp.string cimport string
from cython.operator cimport dereference as deref

# Forward declaraction
cdef class PyCEUpdater:
    cdef CEUpdater* _cpp_class


cdef class PyClusterTracker:
    cdef ClusterTracker *_clust_track

    def __cinit__(self):
        self._clust_track = NULL

    def __init__(self, PyCEUpdater upd, string cname, string element):
        self._clust_track = new ClusterTracker(deref(upd._cpp_class), cname, element)

    def __dealloc__(PyClusterTracker self):
        if self._clust_track != NULL:
            del self._clust_track

    def find_clusters(self):
        self._clust_track.find_clusters()

    def get_cluster_statistics_python(self):
        return self._clust_track.get_cluster_statistics_python()

    def atomic_clusters2group_indx_python(self):
        return self._clust_track.get_cluster_statistics_python()

    def grow_cluster(self, size):
        self._clust_track.grow_cluster(size)

    def surface_python(self):
        self._clust_track.surface_python()
