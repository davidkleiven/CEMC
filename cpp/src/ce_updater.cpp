#include "ce_updater.hpp"
#include <iostream>
#include "numpy/ndarrayobject.h"
#include "additional_tools.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <omp.h>
#include <cassert>
#include <stdexcept>
#include <iterator>

#define CE_DEBUG
using namespace std;

CEUpdater::CEUpdater(){};
CEUpdater::~CEUpdater()
{
  delete history;
  delete vibs; vibs=nullptr;
  if ( atoms != nullptr ) Py_DECREF(atoms);

  for ( unsigned int i=0;i<observers.size();i++ )
  {
    delete observers[i];
  }
}

void CEUpdater::init( PyObject *BC, PyObject *corrFunc, PyObject *pyeci, PyObject *perms )
{
  import_array();
  #ifdef CE_DEBUG
    cerr << "Getting symbols from BC object\n";
  #endif
  // Initialize the symbols
  atoms = PyObject_GetAttrString( BC, "atoms" );
  if ( atoms == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  unsigned int n_atoms = PyObject_Length( atoms );
  for ( unsigned int i=0;i<n_atoms;i++ )
  {
    PyObject *pyindx = PyInt_FromLong(i);
    PyObject *atom = PyObject_GetItem(atoms,pyindx);
    symbols.push_back( PyString_AsString( PyObject_GetAttrString(atom,"symbol")) );
    Py_DECREF(pyindx);
    Py_DECREF(atom);
  }
  trans_symm_group.resize(n_atoms);

  // Build read the translational sites
  PyObject* py_trans_symm_group = PyObject_GetAttrString( BC, "index_by_trans_symm" );
  build_trans_symm_group( py_trans_symm_group );
  Py_DECREF( py_trans_symm_group );

  #ifdef CE_DEBUG
    cerr << "Getting cluster names from atoms object\n";
  #endif

  // Read cluster names
  create_cname_with_dec( corrFunc );
  PyObject *clist = PyObject_GetAttrString( BC, "cluster_names" );
  PyObject *clst_indx = PyObject_GetAttrString( BC, "cluster_indx" );
  if ( clist == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  unsigned int num_trans_symm = PyList_Size( clist );
  /*if ( num_trans_symm != 1 )
  {
    throw invalid_argument( "The CE udpater only supports 1 site type at the moment..." );
  }*/
  // Loop over all symmetry equivalent sites
  for ( unsigned int s=0;s<num_trans_symm;s++ )
  {
    PyObject *clusters_name_list = PyList_GetItem( clist, s );
    PyObject *current_indx_outer = PyList_GetItem( clst_indx, s );
    int n_cluster_sizes = PyList_Size( clusters_name_list );
    map<string,Cluster> new_clusters;
    if ( n_cluster_sizes < 0 )
    {
      throw runtime_error( "Could not read clusters! Length list is smaller than zero!" );
    }

    // Loop over the different cluster sizes
    for ( int i=0;i<n_cluster_sizes;i++ )
    {
      PyObject* current_list_name = PyList_GetItem(clusters_name_list,i);
      PyObject *current_indx_list = PyList_GetItem( current_indx_outer, i );

      int n_clusters = PyList_Size( current_list_name );
      if ( n_clusters < 0 )
      {
        stringstream msg;
        msg << "Could not read clusters for cluster size " << i;
        msg << ". Length is smaller than zero!";
        throw runtime_error( msg.str() );
      }

      // Loop over each cluster at that given size
      for ( int j=0;j<n_clusters;j++ )
      {
        string cluster_name( PyString_AsString( PyList_GetItem(current_list_name,j) ) );
        if ( cname_with_dec.find(cluster_name) == cname_with_dec.end() )
        {
          // Skip this cluster
          continue;
        }
        if ( (cluster_name.substr(0,2) == "c0") || (cluster_name.substr(0,2) == "c1") )
        {
          continue;
        }

        PyObject *py_members = PyList_GetItem( current_indx_list, j );
        int n_sub_clusters = PyList_Size(py_members);
        if ( n_sub_clusters < 0 )
        {
          stringstream msg;
          msg << "Could not read cluster. Size: " << i << " subcluster: " << j << " name: " << cluster_name;
          msg << ". Length smaller than zero!";
          throw runtime_error( msg.str() );
        }

        vector< vector<int> > members;
        // Loop over the indivudal clusters
        for ( int k=0;k<n_sub_clusters;k++ )
        {
          PyObject *py_one_cluster = PyList_GetItem(py_members,k);
          int n_members = PyList_Size(py_one_cluster);
          vector<int> sub_clust;
          for ( unsigned int l=0;l<n_members;l++ )
          {
            sub_clust.push_back( PyInt_AsLong( PyList_GetItem(py_one_cluster,l)) );
          }
          members.push_back(sub_clust);
        }
        if ( cname_with_dec.find(cluster_name) == cname_with_dec.end() )
        {
          stringstream ss;
          ss << "Could not find the full cluster name for name " << cluster_name;
          ss << ". Full cluster names exists for: ";
          ss << cname_with_dec;
          throw invalid_argument( ss.str() );
        }
        string full_cname = cname_with_dec.at(cluster_name);
        new_clusters[cluster_name] = Cluster( cluster_name, members );

        if ( cluster_symm_group_count.find(cluster_name) == cluster_symm_group_count.end() )
        {
          cluster_symm_group_count[cluster_name] = members.size();
        }
        else
        {
          cluster_symm_group_count[cluster_name] += members.size();
        }
      }
    }
    clusters.push_back( new_clusters );
  }
  Py_DECREF( clst_indx );
  Py_DECREF( clist );

  #ifdef CE_DEBUG
    cerr << "Reading basis functions from BC object\n";
  #endif

  PyObject* bfs = PyObject_GetAttrString( BC, "basis_functions" );
  if ( bfs == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  // Reading basis functions from python object
  PyObject *key;
  PyObject *value;
  unsigned int n_bfs = PyList_Size(bfs);
  for ( unsigned int i=0;i<n_bfs;i++ )
  {
    Py_ssize_t pos = 0;
    map<string,double> new_entry;
    PyObject *bf_dict = PyList_GetItem( bfs, i );
    while( PyDict_Next(bf_dict, &pos, &key,&value) )
    {
      new_entry[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
    }
    basis_functions.push_back(new_entry);
  }

  #ifdef CE_DEBUG
    cerr << "Reading translation matrix from BC\n";
  #endif
  PyObject* trans_mat_orig = PyObject_GetAttrString(BC,"trans_matrix");
  if ( trans_mat_orig == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }
  PyObject *trans_mat =  PyArray_FROM_OTF( trans_mat_orig, NPY_INT32, NPY_ARRAY_IN_ARRAY );
  unsigned int max_indx = get_max_indx_of_zero_site(); // Compute the max index that is ever going to be checked

  npy_intp *size = PyArray_DIMS( trans_mat );
  trans_matrix.set_size( size[0],max_indx+1 );
  if ( max_indx+1 > size[1] )
  {
    stringstream ss;
    ss << "Something is wrong with the translation matrix passed.\n";
    ss << "Shape of translation matrix (" << size[0] << "," << size[1] << ")\n";
    ss << "Maximum index encountered in the cluster lists: " << max_indx << endl;
    throw invalid_argument(ss.str());
  }
  for ( unsigned int i=0;i<size[0];i++ )
  for ( unsigned int j=0;j<max_indx+1;j++ )
  {
    trans_matrix(i,j) = *static_cast<int*>(PyArray_GETPTR2(trans_mat,i,j) );
  }

  /**
  trans_matrix.set_size( size[0], size[1] );
  for ( unsigned int i=0;i<size[0];i++ )
  for ( unsigned int j=0;j<size[1];j++ )
  {
    trans_matrix(i,j) = *static_cast<int*>(PyArray_GETPTR2(trans_mat,i,j) );
  }*/

  // Read the ECIs
  Py_ssize_t pos = 0;
  map<string,double> temp_ecis;
  while(  PyDict_Next(pyeci, &pos, &key,&value) )
  {
    temp_ecis[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
  }
  ecis.init(temp_ecis);
  #ifdef CE_DEBUG
    cerr << "Parsing correlation function\n";
  #endif

  vector<string> flattened_cnames;
  flattened_cluster_names(flattened_cnames);
  //history = new CFHistoryTracker(flattened_cnames);
  history = new CFHistoryTracker(ecis.get_names());
  history->insert( corrFunc, nullptr );
  //create_ctype_lookup();
  //create_permutations( perms );

  // Store the singlets names
  for ( unsigned int i=0;i<flattened_cnames.size();i++ )
  {
    if ( flattened_cnames[i].substr(0,2) == "c1" )
    {
      singlets.push_back( flattened_cnames[i] );
    }
  }

  status = Status_t::READY;
  clear_history();
  #ifdef CE_DEBUG
    cout << "CEUpdater initialized sucessfully!\n";
  #endif

  // Verify that the ECIs given corresponds to a correlation function
  if ( !all_eci_corresponds_to_cf() )
  {
    throw invalid_argument( "All ECIs does not correspond to a correlation function!" );
  }
}

void CEUpdater::create_ctype_lookup()
{
  for ( unsigned int n=2;n<cluster_names.size();n++ )
  {
    for ( unsigned int ctype=0;ctype<cluster_names[0][n].size();ctype++ )
    {
      ctype_lookup[cluster_names[0][n][ctype]] = ctype;
    }
  }
}

void CEUpdater::create_permutations( PyObject *perms)
{
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while ( PyDict_Next(perms, &pos, &key, &value) )
  {
    vector< vector<int> > new_vec;
    int size = PyList_Size(value);
    for ( int i=0;i<size;i++ )
    {
      vector<int> one_perm;
      PyObject *cur = PyList_GetItem(value,i);
      int n_entries = PyTuple_Size(cur);
      for ( int j=0;j<n_entries;j++ )
      {
        one_perm.push_back( PyInt_AsLong(PyTuple_GetItem(cur,j) ) );
      }
      new_vec.push_back(one_perm);
    }
    permutations[PyInt_AsLong(key)] = new_vec;
  }
}

double CEUpdater::get_energy()
{
  double energy = 0.0;
  cf& corr_func = history->get_current();
  energy = ecis.dot( corr_func );
  /*
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    energy += corr_func[iter->first]*iter->second;
  }*/
  return energy*symbols.size();
}

double CEUpdater::spin_product_one_atom( unsigned int ref_indx, const vector< vector<int> > &indx_list, const vector<int> &dec, const vector<string> &symbs )
{
  unsigned int num_indx = indx_list.size();
  double sp = 0.0;
  for ( unsigned int i=0;i<num_indx;i++ )
  {
    double sp_temp = 1.0;
    unsigned int n_memb = indx_list[i].size();
    for ( unsigned int j=0;j<n_memb;j++ )
    {
      unsigned int trans_indx = trans_matrix( ref_indx,indx_list[i][j] );
      sp_temp *= basis_functions[dec[j+1]][symbs[trans_indx]];
    }
    sp += sp_temp;
  }
  return sp;
}

void CEUpdater::update_cf( PyObject *single_change )
{
  SymbolChange symb_change;
  py_tuple_to_symbol_change( single_change, symb_change );
  update_cf( symb_change );
}

SymbolChange& CEUpdater::py_tuple_to_symbol_change( PyObject *single_change, SymbolChange &symb_change )
{
  symb_change.indx = PyInt_AsLong( PyTuple_GetItem(single_change,0) );
  symb_change.old_symb = PyString_AsString( PyTuple_GetItem(single_change,1) );
  symb_change.new_symb = PyString_AsString( PyTuple_GetItem(single_change,2) );
  return symb_change;
}

void CEUpdater::update_cf( SymbolChange &symb_change )
{
  if ( symb_change.old_symb == symb_change.new_symb )
  {
    return;
  }

  SymbolChange *symb_change_track;
  cf &current_cf = history->get_current();
  cf *next_cf_ptr=nullptr;
  history->get_next( &next_cf_ptr, &symb_change_track );
  cf &next_cf = *next_cf_ptr;
  symb_change_track->indx = symb_change.indx;
  symb_change_track->old_symb = symb_change.old_symb;
  symb_change_track->new_symb = symb_change.new_symb;
  symb_change_track->track_indx = symb_change.track_indx;



  symbols[symb_change.indx] = symb_change.new_symb;
  if ( atoms != nullptr )
  {
    PyObject *symb_str = PyString_FromString(symb_change.new_symb.c_str());
    PyObject *pyindx = PyInt_FromLong(symb_change.indx);
    PyObject* atom = PyObject_GetItem(atoms, pyindx);
    PyObject_SetAttrString( atom, "symbol", symb_str );
    Py_DECREF(symb_str);
    Py_DECREF(pyindx);
    Py_DECREF(atom);
  }

  // Loop over all ECIs
  //for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  //#pragma omp parallel for
  for ( unsigned int i=0;i<ecis.size();i++ )
  {
    //const string &name = iter->first;
    const string& name = ecis.name(i);
    if ( name.find("c0") == 0 )
    {
      //next_cf[name] = current_cf[name];
      next_cf[i] = current_cf[i];
      continue;
    }

    vector<int> bfs;
    get_basis_functions( name, bfs );
    if ( name.find("c1") == 0 )
    {
      int dec = bfs[0];
      //next_cf[name] = current_cf[name] + (basis_functions[dec][symb_change.new_symb] - basis_functions[dec][symb_change.old_symb])/symbols.size();
      next_cf[i] = current_cf[i] + (basis_functions[dec][symb_change.new_symb] - basis_functions[dec][symb_change.old_symb])/symbols.size();
      continue;
    }

    // Extract the prefix
    int pos = name.rfind("_");
    string prefix = name.substr(0,pos);

    double delta_sp = 0.0;
    int symm = trans_symm_group[symb_change.indx];
    if ( clusters[symm].find(prefix) == clusters[symm].end() )
    {
      next_cf[i] = current_cf[i];
      continue;
    }
    const vector< vector<int> > &cluster_indices = clusters[symm].at(prefix).get();
    unsigned int size = clusters[symm].at(prefix).size;
    double normalization = cluster_indices.size();
    assert( cluster_indices[0].size() == size );
    assert( bfs.size() == size );

    int permutation_counter = 0;
    do
    {
      double sp_ref = spin_product_one_atom( symb_change.indx, cluster_indices, bfs, symbols );
      double sp_new = spin_product_one_atom( symb_change.indx, cluster_indices, bfs, symbols );
      int bf_ref = bfs[0];
      delta_sp += basis_functions[bf_ref][symb_change.new_symb]*sp_new - basis_functions[bf_ref][symb_change.old_symb]*sp_ref;
      permutation_counter += 1;
    } while ( next_permutation( bfs.begin(),bfs.end() ) );
    delta_sp *= (static_cast<double>(size)/permutation_counter);
    //delta_sp /= (normalization*symbols.size()); // This was the old normalization
    delta_sp /= (cluster_symm_group_count.at(prefix)*trans_symm_group_count[symm]);
    //cout << name << " " << cluster_indices << endl;
    next_cf[i] = current_cf[i] + delta_sp;
  }
}

void CEUpdater::undo_changes()
{
  if ( tracker != nullptr )
  {
    undo_changes_tracker();
    return;
  }

  unsigned int buf_size = history->history_size();
  SymbolChange *last_changes;
  for ( unsigned int i=0;i<buf_size-1;i++ )
  {
    history->pop( &last_changes );
    //cout <<"Undo changing " << last_changes->indx << " from " << symbols[last_changes->indx] << " to " << last_changes->old_symb << endl;
    symbols[last_changes->indx] = last_changes->old_symb;

    if ( atoms != nullptr )
    {
      PyObject *old_symb_str = PyString_FromString(last_changes->old_symb.c_str());
      PyObject *pyindx = PyInt_FromLong(last_changes->indx);
      PyObject *pysymb = PyObject_GetItem(atoms, pyindx);
      PyObject_SetAttrString( pysymb, "symbol", old_symb_str );

      // Remove temporary objects
      Py_DECREF(old_symb_str);
      Py_DECREF(pyindx);
      Py_DECREF(pysymb);
    }
  }
}

void CEUpdater::undo_changes_tracker()
{
  //cout << "Undoing changes, keep track\n";
  SymbolChange *last_change;
  SymbolChange *first_change;
  tracker_t& trk = *tracker;
  while( history->history_size() > 1 )
  {
    history->pop(&last_change);
    history->pop(&first_change);
    symbols[last_change->indx] = last_change->old_symb;
    symbols[first_change->indx] = first_change->old_symb;
    trk[first_change->old_symb][first_change->track_indx] = first_change->indx;
    trk[last_change->old_symb][last_change->track_indx] = last_change->indx;
  }
  symbols[first_change->indx] = first_change->old_symb;
  symbols[last_change->indx] = last_change->old_symb;
  //cerr << "History cleaned!\n";
  //cerr << history->history_size() << endl;
}

double CEUpdater::calculate( PyObject *system_changes )
{

  int size = PyList_Size(system_changes);
  if ( size == 1 )
  {
    for ( int i=0;i<size;i++ )
    {
      update_cf( PyList_GetItem(system_changes,i) );
    }
    return get_energy();
  }
  else if ( size == 2 )
  {
    array<SymbolChange,2> changes;
    py_tuple_to_symbol_change( PyList_GetItem(system_changes,0), changes[0] );
    py_tuple_to_symbol_change( PyList_GetItem(system_changes,1), changes[1] );
    return calculate(changes);
  }
  else
  {
    throw runtime_error( "Swaps of more than 2 atoms is not supported!" );
  }
}

double CEUpdater::calculate( array<SymbolChange,2> &system_changes )
{
  if ( symbols[system_changes[0].indx] == symbols[system_changes[1].indx] )
  {
    cout << system_changes[0] << endl;
    cout << system_changes[1] << endl;
    throw runtime_error( "This version of the calculate function assumes that the provided update is swapping two atoms\n");
  }

  if ( symbols[system_changes[0].indx] != system_changes[0].old_symb )
  {
    throw runtime_error( "The atom position tracker does not match the current state\n" );
  }
  else if ( symbols[system_changes[1].indx] != system_changes[1].old_symb )
  {
    throw runtime_error( "The atom position tracker does not match the current state\n" );
  }

  // Update correlation function
  update_cf( system_changes[0] );
  update_cf( system_changes[1] );
  if ( tracker != nullptr )
  {
    tracker_t& trk = *tracker;
    trk[system_changes[0].old_symb][system_changes[0].track_indx] = system_changes[1].indx;
    trk[system_changes[1].old_symb][system_changes[1].track_indx] = system_changes[0].indx;
  }

  return get_energy();
}

void CEUpdater::clear_history()
{
  history->clear();
}

void CEUpdater::flattened_cluster_names( vector<string> &flattened )
{
  /*
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    flattened.push_back( iter->first );
  }*/
  flattened = ecis.get_names();

  // Sort the cluster names for consistency
  sort( flattened.begin(), flattened.end() );
}

PyObject* CEUpdater::get_cf()
{
  PyObject* cf_dict = PyDict_New();
  cf& corrfunc = history->get_current();

  //for ( auto iter=corrfunc.begin(); iter != corrfunc.end(); ++iter )
  for ( unsigned int i=0;i<corrfunc.size();i++ )
  {
    PyObject *pyvalue =  PyFloat_FromDouble(corrfunc[i]);
    PyDict_SetItemString( cf_dict, corrfunc.name(i).c_str(), pyvalue );
    Py_DECREF(pyvalue);
  }
  return cf_dict;
}

CEUpdater* CEUpdater::copy() const
{
  CEUpdater* obj = new CEUpdater();
  obj->symbols = symbols;
  obj->cluster_names = cluster_names;
  obj->cluster_indx = cluster_indx;
  obj->clusters = clusters;
  obj->trans_symm_group = trans_symm_group;
  obj->trans_symm_group_count = trans_symm_group_count;
  obj->cluster_symm_group_count = cluster_symm_group_count;
  obj->basis_functions = basis_functions;
  obj->status = status;
  obj->trans_matrix = trans_matrix;
  obj->ctype_lookup = ctype_lookup;
  obj->ecis = ecis;
  obj->cname_with_dec = cname_with_dec;
  obj->history = new CFHistoryTracker(*history);
  obj->permutations = permutations;
  obj->atoms = nullptr; // Left as nullptr by intention
  obj->tracker = tracker;
  return obj;
}

void CEUpdater::set_symbols( const vector<string> &new_symbs )
{
  if ( new_symbs.size() != symbols.size() )
  {
    throw runtime_error( "The number of atoms in the updater cannot be changed via the set_symbols function\n");
  }
  symbols = new_symbs;
}

void CEUpdater::set_ecis( PyObject *new_ecis )
{
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while( PyDict_Next(new_ecis, &pos, &key,&value) )
  {
    ecis[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
  }

  if ( !all_eci_corresponds_to_cf() )
  {
    throw invalid_argument( "All ECIs has to correspond to a correlation function!" );
  }
}

int CEUpdater::get_decoration_number( const string &cname ) const
{
  if ( basis_functions.size() == 1 )
  {
    return 0;
  }

  // Find position of the last under score
  size_t found = cname.find_last_of("_");
  return atoi( cname.substr(found+1).c_str() )-1;
}

bool CEUpdater::all_decoration_nums_equal( const vector<int> &dec_nums ) const
{
  for ( unsigned int i=1;i<dec_nums.size();i++ )
  {
    if ( dec_nums[i] != dec_nums[0] )
    {
      return false;
    }
  }
  return true;
}

void CEUpdater::get_singlets( PyObject *npy_obj ) const
{
  PyObject *npy_array = PyArray_FROM_OTF( npy_obj, NPY_DOUBLE, NPY_OUT_ARRAY );
  if ( PyArray_SIZE(npy_array) < singlets.size() )
  {
    string msg("The passed Numpy array is too small to hold all the singlets terms!\n");
    stringstream ss;
    ss << "Minimum size: " << singlets.size() << ". Given size: " << PyArray_SIZE(npy_array);
    msg += ss.str();
    Py_DECREF( npy_array );
    throw runtime_error( msg );
  }
  cf& cfs = history->get_current();
  for ( unsigned int i=0;i<singlets.size();i++ )
  {
    double *ptr = static_cast<double*>( PyArray_GETPTR1(npy_array,i) );
    *ptr = cfs[singlets[i]];
  }
  Py_DECREF( npy_array );
}

void CEUpdater::add_linear_vib_correction( const map<string,double> &eci_per_kbT )
{
  delete vibs;
  vibs = new LinearVibCorrection(eci_per_kbT);
}

double CEUpdater::vib_energy( double T ) const
{
  if ( vibs != nullptr )
  {
    cf& corrfunc = history->get_current();
    return vibs->energy( corrfunc, T );
  }
  return 0.0;
}

void CEUpdater::get_basis_functions( const string &cname, vector<int> &bfs ) const
{
  int pos = cname.rfind("_");
  string bfs_str = cname.substr(pos+1);
  bfs.clear();
  for ( unsigned int i=0;i<bfs_str.size();i++ )
  {
    bfs.push_back( bfs_str[i]-'0' );
  }
}

void CEUpdater::create_cname_with_dec( PyObject *cf )
{
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while(  PyDict_Next(cf, &pos, &key,&value) )
  {
    string new_key = PyString_AsString(key);
    if ( new_key.substr(0,2) == "c1" )
    {
      cname_with_dec[new_key] = new_key;
    }
    else
    {
      int pos = new_key.rfind("_");
      string prefix = new_key.substr(0,pos);
      cname_with_dec[prefix] = new_key;
    }
  }
}

void CEUpdater::build_trans_symm_group( PyObject *py_trans_symm_group )
{
  // Fill the symmetry group array with -1 indicating an invalid value
  for ( unsigned int i=0;i<trans_symm_group.size();i++ )
  {
    trans_symm_group[i] = -1;
  }

  int list_size = PyList_Size( py_trans_symm_group );
  for ( int i=0;i<list_size;i++ )
  {
    PyObject *sublist = PyList_GetItem( py_trans_symm_group, i );
    int n_sites = PyList_Size( sublist );
    for ( unsigned int j=0;j<n_sites;j++ )
    {
      int indx = PyInt_AsLong( PyList_GetItem( sublist, j ) );
      if ( trans_symm_group[indx] != -1 )
      {
        throw runtime_error( "One site appears to be present in more than one translation symmetry group!" );
      }
      trans_symm_group[indx] = i;
    }
  }

  // Check that all sites belongs to one translational symmetry group
  for ( unsigned int i=0;i<trans_symm_group.size();i++ )
  {
    if ( trans_symm_group[i] == -1 )
    {
      stringstream msg;
      msg << "Site " << i << " has not been assigned to any translational symmetry group!";
      throw runtime_error( msg.str() );
    }
  }

  // Count the number of atoms in each symmetry group
  trans_symm_group_count.resize(list_size);
  for ( unsigned int i=0;i<trans_symm_group.size();i++ )
  {
    trans_symm_group_count[trans_symm_group[i]] += 1;
  }
}

bool CEUpdater::all_eci_corresponds_to_cf()
{
    cf& corrfunc = history->get_current();
    return ecis.names_are_equal(corrfunc);
}

unsigned int CEUpdater::get_max_indx_of_zero_site() const
{
  unsigned int max_indx = 0;
  // Loop over cluster sizes
  for ( auto iter=clusters.begin(); iter != clusters.end(); ++iter )
  {
    for ( auto subiter=iter->begin(); subiter != iter->end(); ++subiter )
    {
      const vector <vector<int> >& mems = subiter->second.get();
      // Loop over clusters
      for ( unsigned int i=0;i<mems.size();i++ )
      {
        // Loop over members in subcluster
        for ( unsigned int j=0;j<mems[i].size();j++ )
        {
          if ( mems[i][j] > max_indx )
          {
            max_indx = mems[i][j];
          }
        }
      }
    }
  }
  return max_indx;
}
