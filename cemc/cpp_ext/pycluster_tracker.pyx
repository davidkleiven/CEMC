# distutils: language = c++
# distutils: sources =  cpp/src/cluster_tracker.cpp

from cemc.cpp_ext.cluster_tracker cimport ClusterTracker
from cemc.cpp_ext.ce_updater cimport CEUpdater
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef class PyClusterTracker:
    cdef ClusterTracker *_clust_track

    def __cinit__(self):
        self._clust_track = NULL

    def __init__(self, PyCEUpdater upd, vector[string] cname, vector[string] element):
        self._clust_track = new ClusterTracker(deref(upd._cpp_class), cname, element)

    def __dealloc__(PyClusterTracker self):
        if self._clust_track != NULL:
            del self._clust_track

    def find_clusters(self):
        self._clust_track.find_clusters()

    def get_cluster_statistics_python(self):
        return self._clust_track.get_cluster_statistics_python()

    def atomic_clusters2group_indx_python(self):
        return self._clust_track.atomic_clusters2group_indx_python()

    def surface_python(self):
        self._clust_track.surface_python()

    def move_creates_new_cluster(self, system_changes):
        return self._clust_track.move_creates_new_cluster(system_changes)

    def update_clusters(self, system_changes):
        self._clust_track.update_clusters(system_changes)
