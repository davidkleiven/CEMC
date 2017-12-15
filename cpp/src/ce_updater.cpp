#include "ce_updater.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#define CE_DEBUG
using namespace std;

CEUpdater::CEUpdater( PyObject *BC, PyObject *currFunc, PyObject *pyeci )
{
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

  unsigned int n_atoms = PyList_Size( atoms );
  for ( unsigned int i=0;i<n_atoms;i++ )
  {
    PyObject *atom = PyList_GetItem(atoms,i);
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
    vector< vector<int> > outer_list;
    PyObject *current_list = PyList_GetItem( clst_indx, i );
    unsigned int n_clusters = PyList_Size( current_list );
    for ( unsigned int j=0;j<n_clusters;j++ )
    {
      PyObject *members = PyList_GetItem( current_list, j );
      unsigned int n_members = PyList_Size(members);
      vector<int> inner_list;
      for ( unsigned int k=0;k<n_members;k++ )
      {
        inner_list.push_back( PyInt_AsLong( PyList_GetItem(members,k)) );
      }
      outer_list.push_back(inner_list);
    }
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
  unsigned int n_decs = PyList_Size(bfs);
  for ( unsigned int i=0;i<n_decs;i++ )
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
  PyObject *trans_mat =  PyArray_FROM_OTF( PyObject_GetAttrString(BC,"trans_matrix"), NPY_INT32, NPY_ARRAY_IN_ARRAY );
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

  create_ctype_lookup();
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

double CEUpdater::get_energy() const
{
  double energy = 0.0;
  for ( auto iter=ecis.begin(); iter != ecis.end(); ++iter )
  {
    energy += (*corr_functions)[iter->first]*iter->second;
  }
  return energy;
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

void CEUpdater::update_cf( unsigned int indx, const string& old_symb, const string& new_symb )
{
  
}
