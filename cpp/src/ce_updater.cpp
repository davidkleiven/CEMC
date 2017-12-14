#include <Python.h>
#include "ce_updater.hpp"
using namespace std;

CEUpdater updater; // Global object

/**
* Initialize the updater object
*/
static *PyObject init_updater( PyObject *self, PyObject *args )
{
  PyObject *symbols = nullptr;
  PyObject *cluster_names = nullptr;
  PyObject *cluster_indx = nullptr;
  PyObject *basis_functions = nullptr;
  PyObject *cfs = nullptr;

  if ( !PyArg_ParseTuple( args, "OOOOO", &symbols, &cluster_names, &cluster_indx, &basis_functions, &cfs) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the supplied arguments!" )
    return NULL;
  }

  // Insert symbols
  int size = PyList_Size( symbols );
  for ( unsigned int i=0;i<size;i++ )
  {
    updater.symbols.push_back( PyString_AsString( PyList_GetItem(symbols), i) );
  }

  // Insert cluster_names
  int n_cluster_sizes = PyList_Size( cluster_names );
  for ( unsigned int i=0;i<n_cluster_sizes;i++ )
  {
    vector<string> new_list;
    PyObject* current_list = PyList_GetItem(cluster_names,i);
    unsigned int n_clusters = PyList_Size( current_list );
    for ( unsigned int j=0;j<n_clusters;j++ )
    {
      new_list.push_back( PyString_AsString( PyList_GetItem(current_list,j)) );
    }
    updater.cluster_names.push_back(new_list);
  }
}
