#include "wang_landau_sampler.hpp"
#include <omp.h>
using namespace std;

unsigned int num_threads = omp_get_num_threads();

WangLandauSampler::WangLandauSampler( const CEUpdater &updater, PyObject *py_wl )
{
  for ( unsigned int i=0;i<num_threads;i++ )
  {
    updaters.push_back( updater.copy() );
  }

  // Read the symbol position array from the py_object
  PyObject *py_symbpos = PyObject_GetAttrString( py_wl, "atom_positions_track" );
  if ( py_symbpos == nullptr )
  {
    ready = false;
    return;
  }

  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while ( PyDict_Next(py_symbpos, &pos, &key, &value) )
  {
    vector<int> pos;
    int size = PyList_Size(value);
    for ( int i=0;i<size;i++ )
    {
      pos.push_back( PyInt_AsLong(PyList_GetItem(value,i)) );
    }
    atom_positions_track[PyString_AsString(key)] = pos;
  }
  Py_DECREF(py_symbpos);

  PyObject *py_fmin = PyObject_GetAttrString( py_wl, "fmin" );
  min_f = PyFloat_AS_DOUBLE(py_fmin);
  Py_DECREF( py_fmin );

  PyObject *current_f = PyObject_GetAttrString( py_wl, "f" );
  f = PyFloat_AS_DOUBLE( current_f );
  Py_DECREF( current_f );

  PyObject* py_flat = PyObject_GetAttrString( py_wl, "flatness_criteria" );
  flatness_criteria = PyFloat_AS_DOUBLE(py_flat);
  Py_DECREF(py_flat);

  PyObject *py_conv = PyObject_GetAttrString( py_wl, "check_convergence_every" );
  check_convergence_every = PyInt_AsLong( py_conv );
  Py_DECREF(py_conv );

  PyObject *py_site_types = PyObject_GetAttrString( py_wl, "site_types" );
  int size = PyList_Size(py_site_types);
  for ( int i=0;i<size;i++ )
  {
    site_types.push_back( PyInt_AsLong( PyList_GetItem(py_site_types,i)) );
  }
  Py_DECREF( py_site_types );

  PyObject *py_symbols = PyObject_GetAttrString( py_wl, "symbols" );
  size = PyList_Size( py_symbols );
  for ( int i=0;i<size;i++ )
  {
    symbols.push_back( PyString_AsString(PyList_GetItem(py_symbols,i)) );
  }
  Py_DECREF(py_symbols);
}

WangLandauSampler::~WangLandauSampler()
{
  for ( unsigned int i=0;i<updaters.size();i++ )
  {
    delete updaters[i];
  }
}
