#ifndef CLUSTER_TRACKER_H
#define CLUSTER_TRACKER_H
#include <vector>
#include <set>
#include <string>
#include <map>
#include <Python.h>
#include "ce_updater.hpp"

typedef std::vector<std::string> vecstr;
class ClusterTracker
{
public:
  ClusterTracker(CEUpdater &updater, const vecstr &cnames, const vecstr &elements);

  /** Search through the symbol list of CEupdater and identifies atomic clusters */
  void find_clusters();

  /** Collect the cluster statistics */
  void get_cluster_statistics( std::map<std::string,double> &res, std::vector<int> &cluster_sizes ) const;
  PyObject* get_cluster_statistics_python() const;

  /** Converts the atomis-cluster indices into group index */
  void atomic_clusters2group_indx( std::vector<int> &grp_indx ) const;
  PyObject* atomic_clusters2group_indx_python() const;

  /** Verifies that the cluster name provided exists */
  void verify_cluster_name_exists() const;

  /** Get all the members of the largest cluster */
  void get_members_of_largest_cluster( std::vector<int> &members );

  /** Returns a map with all the cluster sizes. Key is the root index */
  void get_cluster_size( std::map<int,int> &cluster_sizes ) const;

  /** Get the root index of the largest cluster */
  unsigned int root_indx_largest_cluster() const;

  /** Returns the root index of an atom */
  unsigned int root_indx( unsigned int indx ) const;

  /** Computes the surface of the clusters */
  void surface( std::map<int,int> &surf ) const;

  /** Check if the proposed move creates a new cluster */
  bool move_creates_new_cluster(PyObject *system_changes);
  bool move_creates_new_cluster(const std::vector<SymbolChange> &system_changes);

  /** Update the clusters */
  void update_clusters(PyObject *system_changes);
  void update_clusters(const std::vector<SymbolChange> &system_changes);

  /** Raise error if circular connected clusters are present */
  void check_circular_connected_clusters();

  /** Return True if indx1 is connected to indx2 */
  bool is_connected(unsigned int indx1, unsigned int indx2) const;

  /** Return true if all atoms have a direct connection to one of its neighbours */
  bool has_minimal_connectivity() const;

  /** Compute the surface of the clusters and return the result in a Python dict */
  PyObject* surface_python() const;
private:
  vecstr elements;
  vecstr cnames;
  CEUpdater *updater; // Do not own this
  std::vector<int> atomic_clusters;
  std::set<int> indices_in_cluster;
  std::vector<std::string> symbols_cpy;

  /** Check if the curent element is one of the cluster elements */
  bool is_cluster_element(const std::string &elm) const;

  /** Detach neighbours from ref_indx. Return True on success. */
  bool detach_neighbours(int ref_indx, bool can_create_new_clusters, const std::vector<int> &indx_in_change);

  /** Initialize indices in cluster */
  void init_cluster_indices();

  /** Rebuild cluster and connect all nodes to one of their neighbours */
  void rebuild_cluster();
};
#endif
