#ifndef CLUSTER_TRACKER_H
#define CLUSTER_TRACKER_H
#include <vector>
#include <string>

class ClusterTracker
{
public:
  ClusterTracker( CEUpdater &updater, const std::string &cname, std::string &element );

  /** Search through the symbol list of CEupdater and identifies atomic clusters */
  void find_clusters();

  /** Collect the cluster statistics */
  std::map<std::string,double> & get_cluster_statistics( std::map<std::string,double> &res );

private:
  std::string element;
  std::string cname;
  CEUpdater *updater; // Not owing
  std::vector<int> atomic_clusters;
};
#endif
