#include "cluster_tracker.hpp"
#include "cluster.hpp"
#include "matrix.hpp"

using namespace std;

ClusterTracker::ClusterTracker( CEUpdater &updater, const std::string &cname, std::string &element ): \
updater(&updater),cname(cname),element(element){};

void ClusterTracker::find_clusters()
{
  const vector<string> &symbs = updater->get_symbols();
  atomic_clusters.resize( symbs.size() );
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

void ClusterTracker::get_cluster_statistics( map<string,double> &res ) const
{
  double average_size = 0.0;
  double max_size = 0.0;
  double avg_size_sq = 0.0;
  map<int,int> num_members_in_cluster;

  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    int counter = 0;
    int root_indx = i;
    vector<int> new_cluster;
    while ( atomic_clusters[root_indx] != -1 )
    {
      counter += 1;
      root_indx = atomic_clusters[root_indx];
    }

    if ( counter > 0 )
    {
      if ( num_members_in_cluster.find(root_indx) != num_members_in_cluster.end() )
      {
        num_members_in_cluster[root_indx] += 1;
      }
      else
      {
        num_members_in_cluster[root_indx] = 1;
      }
    }
  }

  for ( auto iter=num_members_in_cluster.begin(); iter != num_members_in_cluster.end(); ++iter )
  {
    average_size += iter->second;
    avg_size_sq += iter->second*iter->second;
    if ( iter->second > max_size )
    {
      max_size = iter->second;
    }
  }
  res["average_size"] = average_size;
  res["max_size"] = max_size;
  res["avg_size_sq"] = avg_size_sq;
  res["number_of_clusters"] = num_members_in_cluster.size();
}

void ClusterTracker::get_cluster_statistics( PyObject* dict ) const
{
  map<string,int> res,
  get_cluster_statistics(res);
  for ( auto iter=res.begin(); iter != res.end(); ++iter )
  {
    PyObject* value = PyFloat_FromDouble( iter->second );
    PyDict_SetItemString( dict, iter->first.c_str(), value );
    Py_DECREF(value);
  }
}

void ClusterTracker::atomic_clusters2group_indx( vector<int> &group_indx ) const
{
  group_indx.resize( atomic_clusters.size() );
  for ( unsigned i=0;i<atomic_clusters.size();i++ )
  {
    int root_indx = i;
    while ( root_indx != -1 )
    {
      root_indx = atomic_clusters[root_indx];
    }
    group_indx[i] = root_indx;
  }
}

void ClusterTracker::atomic_clusters2group_indx( PyObject *list ) const
{
  vector<int> grp_indx;
  atomic_clusters2group_indx(grp_indx);
  for ( unsigned int i=0;i<grp_indx.size();i++ )
  {
    PyObject *pyint = PyInt_FromLong( grp_indx[i] );
    PyList_Append( list, pyint );
    Py_DECREF(pyint);
  }
}
