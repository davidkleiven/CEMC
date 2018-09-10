#include "cluster_tracker.hpp"
#include "cluster.hpp"
#include "matrix.hpp"
#include "additional_tools.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

ClusterTracker::ClusterTracker(CEUpdater &updater, const vecstr &cname, const vecstr &elements): \
updater(&updater),cnames(cname),elements(elements)
{
  verify_cluster_name_exists();
};

void ClusterTracker::find_clusters()
{
  const vector<string> &symbs = updater->get_symbols();
  atomic_clusters.resize( symbs.size() );
  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    atomic_clusters[i] = -1; // All atoms are initially a root site
  }

  const vector< map<string,Cluster> >& clusters = updater->get_clusters();
  const auto& trans_mat = updater->get_trans_matrix();

  for ( unsigned int i=0;i<symbs.size();i++ )
  {
    // If the element does not match, do not do anything
    if ( !is_cluster_element(symbs[i]) )
    {
      continue;
    }

    int current_root_indx = root_indx(i);

    if ( atomic_clusters[current_root_indx] != -1 )
    {
      throw runtime_error( "Something strange happend. ID of root index is not -1!" );
    }

    // Loop over all symmetries
    for ( unsigned int trans_group=0;trans_group<clusters.size();trans_group++ )
    {
      for (const string &cname : cnames )
      {
        if ( clusters[trans_group].find(cname) == clusters[trans_group].end() )
        {
          // Cluster does not exist in this translattional symmetry group
          continue;
        }

        const std::vector< std::vector<int> >& members = clusters[trans_group].at(cname).get();
        for ( int subgroup=0;subgroup<members.size();subgroup++ )
        {
          int indx = trans_mat(i, members[subgroup][0]);

          //if ( symbs[indx] == element )
          if (is_cluster_element(symbs[indx]))
          {
            int root = root_indx(indx);
            if ( root != current_root_indx )
            {
              atomic_clusters[root] = current_root_indx;
            }
          }
        }
      }
    }
  }
}

void ClusterTracker::get_cluster_size( map<int,int> &num_members_in_cluster ) const
{
  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    int root_indx = i;
    while ( atomic_clusters[root_indx] != -1 )
    {
      root_indx = atomic_clusters[root_indx];
    }

    if ( root_indx != i )
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
}

void ClusterTracker::get_cluster_statistics( map<string,double> &res, vector<int> &cluster_sizes ) const
{
  double average_size = 0.0;
  double max_size = 0.0;
  double avg_size_sq = 0.0;
  map<int,int> num_members_in_cluster;
  cluster_sizes.clear();
  get_cluster_size( num_members_in_cluster );


  for ( auto iter=num_members_in_cluster.begin(); iter != num_members_in_cluster.end(); ++iter )
  {
    int size = 0;
    if ( iter->second >= 1 )
    {
      size = iter->second+1;
      cluster_sizes.push_back(size);
    }
    average_size += size;
    avg_size_sq += size*size;
    if ( size > max_size )
    {
      max_size = size;
    }
  }
  res["avg_size"] = average_size;
  res["max_size"] = max_size;
  res["avg_size_sq"] = avg_size_sq;
  res["number_of_clusters"] = cluster_sizes.size();
}

PyObject* ClusterTracker::get_cluster_statistics_python() const
{
  PyObject* dict = PyDict_New();
  map<string,double> res;
  vector<int> cluster_sizes;
  get_cluster_statistics(res,cluster_sizes);
  for ( auto iter=res.begin(); iter != res.end(); ++iter )
  {
      PyObject* value = PyFloat_FromDouble( iter->second );
      PyDict_SetItemString( dict, iter->first.c_str(), value );
      Py_DECREF(value);
  }

  PyObject* size_list = PyList_New(0);
  for ( int i=0; i< cluster_sizes.size();i++ )
  {
    PyObject *value = int2py( cluster_sizes[i] );
    PyList_Append( size_list, value );
    Py_DECREF(value);
  }
  PyDict_SetItemString( dict, "cluster_sizes", size_list );
  Py_DECREF(size_list);

  return dict;
}

void ClusterTracker::atomic_clusters2group_indx( vector<int> &group_indx ) const
{
  group_indx.resize( atomic_clusters.size() );
  for ( unsigned i=0;i<atomic_clusters.size();i++ )
  {
    int root_indx = i;
    while ( atomic_clusters[root_indx] != -1 )
    {
      root_indx = atomic_clusters[root_indx];
    }
    group_indx[i] = root_indx;
  }
}

PyObject* ClusterTracker::atomic_clusters2group_indx_python() const
{
  PyObject *list = PyList_New(0);
  vector<int> grp_indx;
  atomic_clusters2group_indx(grp_indx);
  for ( unsigned int i=0;i<grp_indx.size();i++ )
  {
    PyObject *pyint = int2py( grp_indx[i] );
    PyList_Append( list, pyint );
    Py_DECREF(pyint);
  }
  return list;
}

void ClusterTracker::verify_cluster_name_exists() const
{
  const vector< map<string,Cluster> >& clusters = updater->get_clusters();
  vector<string> all_names;
  for ( unsigned int i=0;i<clusters.size();i++ )
  {
    for (const string &cname : cnames )
    {
      if ( clusters[i].find(cname) == clusters[i].end() )
      {
        for ( auto iter=clusters[0].begin(); iter != clusters[0].end(); ++iter )
        {
          all_names.push_back( iter->first );
        }
        stringstream ss;
        ss << "There are no correlation functions corresponding to the cluster name given!\n";
        ss << "Given: " << cnames << "\n";
        ss << "Available names:\n";
        ss << all_names;
        throw invalid_argument( ss.str() );
      }
    }
  }
}

unsigned int ClusterTracker::root_indx_largest_cluster() const
{
  map<int,int> clust_size;
  get_cluster_size(clust_size);
  int largest_root_indx = 0;
  int largest_size = 0;
  for ( auto iter=clust_size.begin();iter != clust_size.end(); ++iter )
  {
    if ( iter->second > largest_size )
    {
      largest_root_indx = iter->first;
      largest_size = iter->second;
    }
  }
  return largest_root_indx;
}

void ClusterTracker::get_members_of_largest_cluster( vector<int> &members )
{
  unsigned int root_indx_largest = root_indx_largest_cluster();
  members.clear();
  for ( int id=0;id<atomic_clusters.size();id++ )
  {
    if ( root_indx(id) == root_indx_largest )
    {
      members.push_back(id);
    }
  }
}

unsigned int ClusterTracker::root_indx( unsigned int indx ) const
{
  int root = indx;
  while( atomic_clusters[root] != -1 )
  {
    root = atomic_clusters[root];
  }
  return root;
}

void ClusterTracker::surface( map<int,int> &surf ) const
{
  const vector<string>& symbs = updater->get_symbols();
  const vector< map<string,Cluster> >& clusters = updater->get_clusters();
  const auto& trans_mat = updater->get_trans_matrix();

  for ( unsigned int i=0;i<atomic_clusters.size();i++ )
  {
    if ( atomic_clusters[i] != -1 )
    {
      // This site is part of a cluster
      unsigned int root = root_indx(i);
      if ( surf.find(root) == surf.end() )
      {
        surf[root] = 0;
      }

      for ( unsigned int symm_group=0;symm_group<clusters.size();symm_group++ )
      {

        for (const string &cname : cnames )
        {
          if ( clusters[symm_group].find(cname) == clusters[symm_group].end() )
          {
            // Cluster does not exist in this translattional symmetry group
            continue;
          }
          const vector< vector<int> >& members = clusters[symm_group].at(cname).get();
          for ( int subgroup=0;subgroup<members.size();subgroup++ )
          {
            int indx = trans_mat( i,members[subgroup][0] );
            if (!is_cluster_element(symbs[indx]))
            {
              surf[root] += 1;
            }
          }
        }
      }
    }
  }
}

PyObject* ClusterTracker::surface_python() const
{
  map<int,int> surf;
  surface(surf);
  PyObject* dict = PyDict_New();
  for ( auto iter=surf.begin(); iter != surf.end(); ++iter )
  {
    PyObject* py_int_key = int2py(iter->first);
    PyObject* py_int_surf = int2py(iter->second);
    PyDict_SetItem( dict, py_int_key, py_int_surf );
    Py_DECREF(py_int_key);
    Py_DECREF(py_int_surf);
  }
  return dict;
}

bool ClusterTracker::is_cluster_element(const string &element) const
{
  for (const string &item : elements)
  {
    if (item == element) return true;
  }
  return false;
}
