#include "ce_updater.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include "additional_tools.hpp"
#include <iostream>
#include <sstream>
#define CE_DEBUG
using namespace std;

CEUpdater::CEUpdater(){};

void CEUpdater::init( PyObject *BC, PyObject *corrFunc, PyObject *pyeci )
{
  import_array();
  #ifdef CE_DEBUG
    cerr << "Getting symbols from BC object\n";
  #endif
  // Initialize the symbols
  PyObject *atoms = PyObject_GetAttrString( BC, "atoms" );
  if ( atoms == NULL )
  {
    status = Status_t::INIT_FAILED;
    return;
  }

  unsigned int n_atoms = PyObject_Length( atoms );
  for ( unsigned int i=0;i<n_atoms;i++ )
  {
    PyObject *atom = PyObject_GetItem(atoms,PyInt_FromLong(i));
    symbols.push_back( PyString_AsString( PyObject_GetAttrString(atom,"symbol")) );
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
    cerr << n_clusters << endl;
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
    cerr << "Reading translation matrix from BC";
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
  history.insert( corrFunc, nullptr );
  create_ctype_lookup();
  create_permutations();

  status = Status_t::READY;
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

void CEUpdater::create_permutations()
{
  #ifdef CE_DEBUG
    cerr << "Creating lookup for basis function permutations\n";
  #endif
  PyObject* module_name = PyString_FromString("itertools");
  PyObject* itertools = PyImport_Import(module_name);
  PyObject* product = PyObject_GetAttrString(itertools,"product");

  PyObject* bf_indx = PyList_New(basis_functions.size());
  for ( unsigned int i=0;i<basis_functions.size();i++ )
  {
    PyList_SetItem( bf_indx, i, PyInt_FromLong(i) );
  }

  PyObject *keywords = PyDict_New();
  PyObject* args = PyTuple_New(1);
  PyTuple_SetItem( args, 0, bf_indx );
  PyObject* prms;
  for ( unsigned int i=2;i<4;i++ )
  {
    PyDict_SetItemString( keywords, "repeat", PyInt_FromLong(i) );
    PyObject* result = PyObject_Call( product, args, keywords );
    PyArg_ParseTuple(result,"O",&prms);
    cerr << "here\n";
    int size = PyList_Size(prms);
    cerr << size << endl;
    vector< vector<int> > new_vec;
    for ( int j=0;j<size;j++ )
    {
      vector<int> subvec;
      PyObject* current_perm = PyList_GetItem(prms,j);
      int sub_size = PyList_Size(current_perm);
      for ( int k=0;k<sub_size;k++ )
      {
        subvec.push_back( PyInt_AsLong(PyList_GetItem(current_perm,k)) );
      }
      new_vec.push_back(subvec);
    }
    permutations[i] = new_vec;
    Py_DECREF(args);
    Py_DECREF(keywords);
  }

  #ifdef CE_DEBUG
    cerr << "Basis function permutations finished\n";
  #endif
}

double CEUpdater::get_energy()
{
  double energy = 0.0;
  cf& next_cf = history.get_current();
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    energy += next_cf[iter->first]*iter->second;
  }
  return energy;
}

double CEUpdater::spin_product_one_atom( unsigned int ref_indx, const vector< vector<int> > &indx_list, const vector<int> &dec )
{
  cerr << "spin_prod\n";
  unsigned int num_indx = indx_list.size();
  double sp = 0.0;
  cerr << "Cluster index list size: " << indx_list.size() << endl;
  for ( unsigned int i=0;i<num_indx;i++ )
  {
    double sp_temp = 1.0;
    unsigned int n_memb = indx_list[i].size();
    for ( unsigned int j=0;j<n_memb;j++ )
    {
      unsigned int trans_indx = trans_matrix( ref_indx,indx_list[i][j] );
      cerr << trans_indx << endl;
      sp_temp *= basis_functions[dec[j+1]][symbols[trans_indx]];
    }
    sp += sp_temp;
  }
  return sp;
}

void CEUpdater::update_cf( PyObject *single_change )
{
  cf *next_cf_ptr=nullptr;
  PyObject *next_change=nullptr;
  history.get_next( next_cf_ptr, next_change );
  cf &next_cf = *next_cf_ptr;
  next_change = single_change;
  int indx;
  string new_symb;
  indx = PyInt_AsLong( PyTuple_GetItem(single_change,0) );
  new_symb = PyString_AsString( PyTuple_GetItem(single_change,2) );
  string old_symb = symbols[indx];
  cerr << indx << " " << old_symb << " " << new_symb << endl;

  if ( old_symb == new_symb )
  {
    return;
  }

  symbols[indx] = new_symb;
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    const string &name = iter->first;
    if ( name.find("c0") == 0 )
    {
      continue;
    }
    int dec = static_cast<int>( name.back() )-1;
    if ( name.find("c1") == 0 )
    {
      next_cf[name] += (basis_functions[dec][new_symb] - basis_functions[dec][old_symb]);
      continue;
    }

    int pos_last_underscore = name.find_last_of("_");
    string prefix = name.substr(0,pos_last_underscore);
    string size_str = prefix.substr(1,1);
    int size = atoi(size_str.c_str());
    int ctype = ctype_lookup[prefix];
    double normalization = cluster_indx[size][ctype].size()*symbols.size();
    cerr << prefix << " " << prefix[1] << " " << size << " " << ctype << " " << cluster_indx[size].size() << endl;
    double sp = spin_product_one_atom( indx, cluster_indx[size][ctype], permutations[size][dec] );
    int bf_ref = permutations[size][dec][0];
    sp *= size*( basis_functions[bf_ref][new_symb] - basis_functions[bf_ref][old_symb] );
    cerr << "New spin: " << sp << endl;
    sp /= normalization;
    next_cf[name] += sp;
  }
}

void CEUpdater::undo_changes()
{
  unsigned int buf_size = history.history_size();
  PyObject *last_changes;
  for ( unsigned int i=0;i<buf_size;i++ )
  {
    history.pop( last_changes );
    int indx = PyInt_AsLong( PyList_GetItem(last_changes,0) );
    string old_symb = PyString_AsString( PyList_GetItem(last_changes,1) );
    symbols[indx] = old_symb;
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

void CEUpdater::clear_history()
{
  history.clear();
}
