#ifndef CLUSTER_TRACKER_H
#define CLUSTER_TRACKER_H
#include <vector>
#include <string>
#include <map>
#include <Python.h>
#include "ce_updater.hpp"

class ClusterTracker
{
public:
  ClusterTracker( CEUpdater &updater, const std::string &cname, const std::string &element );

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

  /** Grow a cluster of with a certain size */
  void grow_cluster( unsigned int size );

  /** Get all the members of the largest cluster */
  void get_members_of_largest_cluster( std::vector<int> &members );

  /** Returns a map with all the cluster sizes. Key is the root index */
  void get_cluster_size( std::map<int,int> &cluster_sizes ) const;

  /** Get the root index of the largest cluster */
  unsigned int root_indx_largest_cluster() const;

  /** Returns the root index of an atom */
  unsigned int root_indx( unsigned int indx ) const;
private:
  std::string element;
  std::string cname;
  CEUpdater *updater; // Do not own this
  std::vector<int> atomic_clusters;
};
#endif
