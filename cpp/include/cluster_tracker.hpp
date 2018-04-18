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
private:
  std::string element;
  std::string cname;
  CEUpdater *updater; // Do not own this
  std::vector<int> atomic_clusters;
};
#endif
