#include "ce_updater.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#define CE_DEBUG
using namespace std;

CEUpdater::CEUpdater( PyObject *BC, PyObject *currFunc )
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
    vector< vector<double> > outer_list;
    PyObject *current_list = PyList_GetItem( clst_indx, i );
    unsigned int n_clusters = PyList_Size( current_list );
    for ( unsigned int j=0;j<n_clusters;j++ )
    {
      PyObject *members = PyList_GetItem( current_list, j );
      unsigned int n_members = PyList_Size(members);
      vector<double> inner_list;
      for ( unsigned int k=0;k<n_members;k++ )
      {
        inner_list.push_back( PyFloat_AS_DOUBLE( PyList_GetItem(members,k)) );
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
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while( PyDict_Next(bfs, &pos, &key,&value) )
  {
    basis_functions[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
  }

  #ifdef CE_DEBUG
    cerr << "Reading translation matrix from BC";
  #endif
  PyObject *trans_mat =  PyArray_FROM_OTF( PyObject_GetAttrString(BC,"trans_matrix"), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  npy_intp *size = PyArray_DIMS( trans_mat );
  trans_matrix.set_size( size[0], size[1] );
  for ( unsigned int i=0;i<size[0];i++ )
  for ( unsigned int j=0;j<size[1];j++ )
  {
    trans_matrix(i,j) = *static_cast<double*>(PyArray_GETPTR2(trans_mat,i,j) );
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
