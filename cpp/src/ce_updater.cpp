#include "ce_updater.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include "additional_tools.hpp"
#include <iostream>
#include <sstream>
#define CE_DEBUG
using namespace std;

CEUpdater::CEUpdater(){};
CEUpdater::~CEUpdater()
{
  delete history;
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

  #ifdef CE_DEBUG
    cerr << "Getting cluster names from atoms object\n";
  #endif

  // Read cluster names
  PyObject *clist = PyObject_GetAttrString( BC, "cluster_names" );
  if ( clist == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  unsigned int n_cluster_sizes = PyList_Size( clist );
  for ( unsigned int i=0;i<n_cluster_sizes;i++ )
  {
    vector<string> new_list;
    PyObject* current_list = PyList_GetItem(clist,i);
    unsigned int n_clusters = PyList_Size( current_list );
    for ( unsigned int j=0;j<n_clusters;j++ )
    {
      new_list.push_back( PyString_AsString( PyList_GetItem(current_list,j)) );
    }
    cluster_names.push_back(new_list);
  }


  #ifdef CE_DEBUG
    cerr << "Getting cluster indices from atoms object\n";
  #endif
  // Read cluster indices
  PyObject *clst_indx = PyObject_GetAttrString( BC, "cluster_indx" );
  if ( clst_indx == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  n_cluster_sizes = PyList_Size( clst_indx );
  for ( unsigned int i=0;i<n_cluster_sizes;i++ )
  {
    vector< vector< vector<int> > > outer_list;
    if ( i <= 1 )
    {
      // Insert empty lists
      cluster_indx.push_back(outer_list);
      continue;
    }

    PyObject *current_list = PyList_GetItem( clst_indx, i );
    int n_clusters = PyList_Size( current_list );
    if ( n_clusters < 0 )
    {
      status = Status_t::INIT_FAILED;
      return;
    }

    for ( int j=0;j<n_clusters;j++ )
    {
      PyObject *members = PyList_GetItem( current_list, j );
      int n_members = PyList_Size(members);
      if ( n_members < 0 )
      {
        status = Status_t::INIT_FAILED;
        return;
      }
      vector< vector<int> > inner_list;
      for ( int k=0;k<n_members;k++ )
      {
        vector<int> one_cluster;
        PyObject *py_one_cluster = PyList_GetItem(members,k);
        int n_members_in_cluster = PyList_Size(py_one_cluster);
        if ( n_members_in_cluster < 0 )
        {
          status = Status_t::INIT_FAILED;
          return;
        }

        for ( int l=0;l<n_members_in_cluster;l++ )
        {
          one_cluster.push_back( PyInt_AsLong( PyList_GetItem(py_one_cluster,l)) );
        }
        inner_list.push_back( one_cluster );
      }
      outer_list.push_back(inner_list);
    }
    cluster_indx.push_back(outer_list);
  }

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


  npy_intp *size = PyArray_DIMS( trans_mat );

  trans_matrix.set_size( size[0], size[1] );
  for ( unsigned int i=0;i<size[0];i++ )
  for ( unsigned int j=0;j<size[1];j++ )
  {
    trans_matrix(i,j) = *static_cast<int*>(PyArray_GETPTR2(trans_mat,i,j) );
  }

  // Read the ECIs
  Py_ssize_t pos = 0;
  while(  PyDict_Next(pyeci, &pos, &key,&value) )
  {
    ecis[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
  }

  #ifdef CE_DEBUG
    cerr << "Parsing correlation function\n";
  #endif

  vector<string> flattened_cnames;
  flattened_cluster_names(flattened_cnames);
  history = new CFHistoryTracker(flattened_cnames);
  history->insert( corrFunc, nullptr );
  create_ctype_lookup();
  create_permutations( perms );

  status = Status_t::READY;
  clear_history();
  #ifdef CE_DEBUG
    cout << "CEUpdater initialized sucessfully!\n";
  #endif
}

void CEUpdater::create_ctype_lookup()
{
  for ( unsigned int n=2;n<cluster_names.size();n++ )
  {
    for ( unsigned int ctype=0;ctype<cluster_names[n].size();ctype++ )
    {
      ctype_lookup[cluster_names[n][ctype]] = ctype;
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
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    energy += corr_func[iter->first]*iter->second;
  }
  return energy*symbols.size();
}

double CEUpdater::spin_product_one_atom( unsigned int ref_indx, const vector< vector<int> > &indx_list, const vector<int> &dec )
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
      sp_temp *= basis_functions[dec[j+1]][symbols[trans_indx]];
    }
    sp += sp_temp;
  }
  return sp;
}

void CEUpdater::update_cf( PyObject *single_change )
{
  SymbolChange symb_change;
  symb_change.indx = PyInt_AsLong( PyTuple_GetItem(single_change,0) );
  symb_change.old_symb = PyString_AsString( PyTuple_GetItem(single_change,1) );
  symb_change.new_symb = PyString_AsString( PyTuple_GetItem(single_change,2) );
  update_cf( symb_change );
}

void CEUpdater::update_cf( SymbolChange &symb_change )
{
  SymbolChange *symb_change_track;
  cf &current_cf = history->get_current();
  cf *next_cf_ptr=nullptr;
  history->get_next( &next_cf_ptr, &symb_change_track );
  cf &next_cf = *next_cf_ptr;
  symb_change_track->indx = symb_change.indx;
  symb_change_track->old_symb = symb_change.old_symb;
  symb_change_track->new_symb = symb_change.new_symb;

  if ( symb_change.old_symb == symb_change.new_symb )
  {
    return;
  }

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

  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    const string &name = iter->first;
    if ( name.find("c0") == 0 )
    {
      continue;
    }
    string dec_str = name.substr(name.size()-1,1);
    int dec = atoi(dec_str.c_str())-1;
    if ( name.find("c1") == 0 )
    {
      next_cf[name] = current_cf[name] + (basis_functions[dec][symb_change.new_symb] - basis_functions[dec][symb_change.old_symb]);
      continue;
    }

    int pos_last_underscore = name.find_last_of("_");
    string prefix = name.substr(0,pos_last_underscore);
    string size_str = prefix.substr(1,1);
    int size = atoi(size_str.c_str());
    int ctype = ctype_lookup[prefix];
    double normalization = cluster_indx[size][ctype].size()*symbols.size();
    double sp = spin_product_one_atom( symb_change.indx, cluster_indx[size][ctype], permutations[size][dec] );
    int bf_ref = permutations[size][dec][0];
    sp *= size*( basis_functions[bf_ref][symb_change.new_symb] - basis_functions[bf_ref][symb_change.old_symb] );
    sp /= normalization;
    next_cf[name] = current_cf[name] + sp;
  }
}

void CEUpdater::undo_changes()
{
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

double CEUpdater::calculate( PyObject *system_changes )
{
  int size = PyList_Size(system_changes);
  for ( int i=0;i<size;i++ )
  {
    update_cf( PyList_GetItem(system_changes,i) );
  }
  return get_energy();
}

double CEUpdater::calculate( array<SymbolChange,2> &system_changes )
{
  update_cf( system_changes[0] );
  update_cf( system_changes[1] );
  return get_energy();
}

void CEUpdater::clear_history()
{
  history->clear();
}

void CEUpdater::flattened_cluster_names( vector<string> &flattened )
{
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    flattened.push_back( iter->first );
  }
}

PyObject* CEUpdater::get_cf()
{
  PyObject* cf_dict = PyDict_New();
  cf& corrfunc = history->get_current();

  for ( auto iter=corrfunc.begin(); iter != corrfunc.end(); ++iter )
  {
    PyObject *pyvalue =  PyFloat_FromDouble(iter->second);
    PyDict_SetItemString( cf_dict, iter->first.c_str(), pyvalue );
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
  obj->basis_functions = basis_functions;
  obj->status = status;
  obj->trans_matrix = trans_matrix;
  obj->ctype_lookup = ctype_lookup;
  obj->ecis = ecis;
  obj->history = new CFHistoryTracker(*history);
  obj->permutations = permutations;
  obj->atoms = nullptr; // Left as nullptr by intention
  return obj;
}
