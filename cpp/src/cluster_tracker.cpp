#include "cluster_tracker.hpp"
#include "clusters.hpp"
#include "matrix.hpp"

using namespace std;

ClusterTracker::ClusterTracker( CEUpdater &updater, const std::string &cname, std::string &element ): \
updater(&updater),cname(cname),elements(element){};

void ClusterTracker::find_clusters()
{
  const vector<string> &symbs = updater->get_symbols();
  atomic_clusters.set_size( symbs.size() );
  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    atomic_clusters[i] = -1; // All atoms are initially a root site
  }
  const vector< map<string,Cluster> >& clusters = updater->get_clusters();
  const Matrix<int>& trans_mat = updater->get_trans_matrix();

  for ( unsigned int i=0;i<symbs.size();i++ )
  {
    // If the element does not match, do not do anything
    if ( symbs[i] != element )
    {
      continue;
    }

    // Loop over all symmetries
    for ( unsigned int trans_group=0;trans_group<clusters.size();trans_group++ )
    {
      if ( clusters[trans_group].find(cname) == clusters[trans_group].end() )
      {
        // Cluster does not exist in this translattional symmetry group
        continue;
      }

      const std::vector< std::vector<int> >& members = clusters[trans_group].at(cname).get();
      for ( int subgroup=0;subgroup<members.size();subgroup++ )
      {
        int indx = trans_mat(i,members[subgroup][0]);

        if ( symbs[indx] == element )
        {
          int root_indx = indx;
          while( atomic_clusters[root_indx] != -1 )
          {
            root_indx = atomic_clusters[root_indx];
          }
          atomic_clusters[root_indx] = i;
        }
      }
    }
  }
}

map<string,double>& ClusterTracker::get_cluster_statistics( map<string,double> &res )
{
  double average_size = 0.0;
  double max_size = 0.0;
  double avg_size_sq = 0.0;
  int overall_number_of_clusters = 0;
  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    int counter = 0;
    int root_indx = i;
    while ( atomic_clusters[root_indx] != -1 )
    {
      counter += 1:
      root_indx = atomic_clusters[root_indx];
    }
    if ( counter > 0 )
    {
      int cluster_size = counter + 1;
      average_size += cluster_size;
      if ( cluster_size > max_size )
      {
        max_size = cluster_size;
      }
      avg_size_sq += cluster_size*cluster_size;
      overall_number_of_clusters += 1;
    }
  }

  res["average_size"] = average_size;
  res["max_size"] = max_size;
  res["avg_size_sq"] = avg_size_sq;
  res["number_of_clusters"] = overall_number_of_clusters;
  return res;
}
