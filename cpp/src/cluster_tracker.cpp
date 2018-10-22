#include "cluster_tracker.hpp"
#include "cluster.hpp"
#include "matrix.hpp"
#include "additional_tools.hpp"
#include <stdexcept>
#include <sstream>
//#define CLUSTER_TRACK_DEBUG

using namespace std;

ClusterTracker::ClusterTracker(CEUpdater &updater, const vecstr &cname, const vecstr &elements): \
updater(&updater),cnames(cname),elements(elements)
{
  verify_cluster_name_exists();
  init_cluster_indices();
  find_clusters();

  #ifdef CLUSTER_TRACK_DEBUG
    cout << "Cluster tracker initialized\n";
  #endif
}

void ClusterTracker::find_clusters()
{
  const vector<string> &symbs = updater->get_symbols();

  // Syncronize the copy
  symbols_cpy = symbs;

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
    int r_indx = root_indx(i);

    if ( r_indx != i )
    {
      if ( num_members_in_cluster.find(r_indx) != num_members_in_cluster.end() )
      {
        num_members_in_cluster[r_indx] += 1;
      }
      else
      {
        num_members_in_cluster[r_indx] = 1;
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
    unsigned int max_cluster_size = atomic_clusters.size();
    unsigned int counter = 0;
  
    while ((atomic_clusters[root_indx] != -1) && (counter < max_cluster_size))
    {
      counter += 1;
      root_indx = atomic_clusters[root_indx];
    }
    group_indx[i] = root_indx;

    if (counter >= max_cluster_size){
      throw runtime_error("Circular connected clusters seems to be present!");
    }
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
  unsigned int counter = 0;
  unsigned int max_cluster_size = atomic_clusters.size();
  while((atomic_clusters[root] != -1) && (counter<max_cluster_size))
  {
    counter += 1;
    root = atomic_clusters[root];
  }

  if (counter >= max_cluster_size){
    throw runtime_error("Circular connected clusters appears to be present!");
  }
  return root;
}

bool ClusterTracker::is_connected(unsigned int indx1, unsigned int indx2) const{
  unsigned int counter = 0;
  unsigned int max_cluster_size = atomic_clusters.size();
  int root = indx1;
  while((atomic_clusters[root] != -1) && (counter<max_cluster_size))
  {
    counter += 1;
    root = atomic_clusters[root];
    if (root == indx2){
      return true;
    }
  }

  if (counter >= max_cluster_size){
    throw runtime_error("Circular connected clusters appears to be present!");
  }
  return false;
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

void ClusterTracker::update_clusters(PyObject *py_changes){
  vector<SymbolChange> changes;
  py_changes2symb_changes(py_changes, changes);
  update_clusters(changes);
}

void ClusterTracker::update_clusters(const vector<SymbolChange> &changes){
  #ifdef CLUSTER_TRACK_DEBUG
    cout << changes << endl;
  #endif

  const auto& trans_mat = updater->get_trans_matrix();
  const vector<string>& symbs = updater->get_symbols();

  vector<int> indx_in_change;
  for (const SymbolChange &change : changes){
    indx_in_change.push_back(change.indx);
  }

  for (const SymbolChange &change : changes){
    if (is_cluster_element(change.old_symb) && is_cluster_element(change.new_symb)){
      // Change only involved elements in the cluster
      // we don't need to do anything
      continue;
    }
    if (is_cluster_element(change.old_symb)){
        // Detach indices that are connected to this cluster
        detach_neighbours(change.indx, true, indx_in_change);

        #ifdef CLUSTER_TRACK_DEBUG
          cout << "Neighbours detached\n";
          check_circular_connected_clusters();
          cout << "No circular connected clusters after detach\n";
        #endif
    }

    if (is_cluster_element(change.new_symb)){
      // Attach the updated index
      bool attached = false;
      for (auto iter=indices_in_cluster.begin(); iter != indices_in_cluster.end(); ++iter){
        int indx = trans_mat(change.indx, *iter);
        
        // Use symbs (and not symbols_cpy) where the move have already been updated
        if (is_cluster_element(symbs[indx])){
          atomic_clusters[change.indx] = indx;
          attached = true;
          break;
        }
      }
      if (!attached){
        // Create a new reference site
        atomic_clusters[change.indx] = -1;
      }

      #ifdef CLUSTER_TRACK_DEBUG
        cout << "Attached symbol to new cluster\n";
      #endif
    }
    else{
      atomic_clusters[change.indx] = -1;
    }
  }

  // Update the local copy
  for (const SymbolChange &change : changes){
    symbols_cpy[change.indx] = change.new_symb;
  }

  #ifdef CLUSTER_TRACK_DEBUG
    check_circular_connected_clusters();
    cout << "Successfully updated clusters\n";

    if (symbols_cpy != symbs){
      throw runtime_error("The local copy of symbols differ from the CEUpdater.symbol!");
    }
  #endif
}

bool ClusterTracker::move_creates_new_cluster(PyObject *py_changes){
  vector<SymbolChange> changes;
  py_changes2symb_changes(py_changes, changes);
  return move_creates_new_cluster(changes);
}

bool ClusterTracker::move_creates_new_cluster(const vector<SymbolChange> &changes){
  const auto& trans_mat = updater->get_trans_matrix();
  const vector<string>& symbs = updater->get_symbols();

  vector<int> indx_in_change;
  for (const SymbolChange &change : changes){
    indx_in_change.push_back(change.indx);
  }

  for (const SymbolChange &change : changes){
    if (is_cluster_element(change.old_symb) && is_cluster_element(change.new_symb)){
      // Change only involved elements in the cluster
      // we don't need to do anything
      continue;
    }

    if (is_cluster_element(change.old_symb)){
      // Check if all neighbours can be connected 
      // to something else
      if (!detach_neighbours(change.indx, false, indx_in_change)) return true;
    }

    if (is_cluster_element(change.new_symb)){
      // See if we can attach the new position to the a cluster
      bool can_attach = false;
      for (auto iter=indices_in_cluster.begin(); iter != indices_in_cluster.end(); ++iter){
        int indx = trans_mat(change.indx, *iter);
        if (is_cluster_element(symbs[indx])){
          can_attach = true;
          break;
        }
      }
      if (!can_attach) return true;
    }
  }
  return false;
}

bool ClusterTracker::detach_neighbours(unsigned int ref_indx, bool can_create_new_clusters, const vector<int> &indx_in_change){
  const auto& trans_mat = updater->get_trans_matrix();
  //const vector<string>& symbs = updater->get_symbols();

  // if (!is_cluster_element(symbs[ref_indx])){
  //   // There is not to detach as this site is not
  //   // part of the cluster
  //   return true;
  // }

  bool new_ref_indx_assigned = false;
  bool is_ref_indx = (atomic_clusters[ref_indx] == -1);

  // Loop accross neighbour indices
  for (auto iter=indices_in_cluster.begin(); iter != indices_in_cluster.end();++iter){
    unsigned int indx = trans_mat(ref_indx, *iter);

    if (!is_cluster_element(symbols_cpy[indx])){
      // This symbol is not part of the cluster
      continue;
    }

    if (atomic_clusters[indx] != ref_indx){
      // Not connected to the ref_index
      continue;
    }

    if (is_ref_indx && !new_ref_indx_assigned){
      // Move the reference index to this site
      atomic_clusters[indx] = -1;

      // Make the current index point to the new ref_indx
      atomic_clusters[ref_indx] = indx;
      new_ref_indx_assigned = true;
      continue;
    }

    // The current index is connected to ref_indx
    bool managed_to_detach = false;
    for (auto iter2=indices_in_cluster.begin();iter2 != indices_in_cluster.end(); ++iter2){
      int indx2 = trans_mat(indx, *iter2);
      if (is_in_vector(indx2, indx_in_change)) continue;

      // indx is in the cluster, and indx2 is not connected to indx, 
      // we can connect indx to indx2
      if (is_cluster_element(symbols_cpy[indx2]) && !is_connected(indx2, indx)){
        atomic_clusters[indx] = indx2;
        managed_to_detach = true;
        break;
      }
    }

    if (!managed_to_detach && can_create_new_clusters){
        // This site becomes a new reference site
        atomic_clusters[indx] = -1;
    }
    else if (!managed_to_detach){
      // We could not detach this site from the cluster
      // without forming a new cluster.
      // This move should not be performed
      return false;
    }
  }
  return true;
}

void ClusterTracker::init_cluster_indices(){
  const vector< map<string,Cluster> >& clusters = updater->get_clusters();
  for ( const map<string,Cluster> &cluster : clusters )
  {
    for (const string &cname : cnames )
    {
      if ( cluster.find(cname) == cluster.end() )
      {
        continue;
      }
      for (const std::vector<int> &indices : cluster.at(cname).get()){
         indices_in_cluster.insert(indices[0]);
      }
    }
  }
}

void ClusterTracker::check_circular_connected_clusters(){
  for (unsigned int ref_indx=0;ref_indx<atomic_clusters.size();ref_indx++){
    root_indx(ref_indx);
  }
}