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
  ClusterTracker( CEUpdater &updater, const std::string &cname, std::string &element );

  /** Search through the symbol list of CEupdater and identifies atomic clusters */
  void find_clusters();

  /** Collect the cluster statistics */
  void get_cluster_statistics( std::map<std::string,double> &res ) const;
  void get_cluster_statistics( PyObject *dict ) const;

  /** Converts the atomis-cluster indices into group index */
  void atomic_clusters2group_indx( std::vector<int> &grp_indx ) const;
  void atomic_clusters2group_indx( PyObject *list ) const;
private:
  std::string element;
  std::string cname;
  CEUpdater *updater; // Do not own this
  std::vector<int> atomic_clusters;
};
#endif
