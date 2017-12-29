#include "histogram.hpp"
#include <numpy/ndarrayobject.h>
#include <iostream>

using namespace std;

Histogram::Histogram( unsigned int Nbins, double Emin, double Emax ):Nbins(Nbins),Emin(Emin),Emax(Emax)
{
  //import_array();
  hist.resize(Nbins);
  logdos.resize(Nbins);
  known_structures.resize(Nbins);

  for ( unsigned int i=0;i<Nbins;i++ )
  {
    hist[i] = 0;
    logdos[i] = 0.0;
    known_structures[i] = false;
  }
}

double Histogram::get_energy( int bin ) const
{
  return Emin + ((Emax-Emin)*bin)/Nbins;
}

int Histogram::get_bin( double energy ) const
{
  return ( energy-Emin )*Nbins/(Emax-Emin);
}

void Histogram::update( unsigned int bin, double mod_factor )
{
  known_structures[bin] = true;

  hist[bin] += 1;
  logdos[bin] += mod_factor;
}

bool Histogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int count = 0;
  for ( unsigned int i=0;i<hist.size();i++ )
  {
    if ( !known_structures[i] ) continue;

    mean += hist[i];
    count += 1;
    if ( hist[i] < minimum ) minimum = hist[i];
  }

  double mean_dbl = static_cast<double>(mean)/count;
  return minimum > criteria*mean_dbl;
}

double Histogram::get_dos_ratio_old_divided_by_new( unsigned int old_bin, unsigned int new_bin ) const
{
  double diff = logdos[old_bin] - logdos[new_bin];
  return exp(diff);
}

bool Histogram::bin_in_range( int bin ) const
{
  return (bin >= 0) && ( bin < hist.size() );
}

void Histogram::send_to_python_hist( PyObject *py_hist )
{
  import_array();
  PyObject *py_visits = PyObject_GetAttrString( py_hist, "histogram" );
  PyObject *py_visits_npy = PyArray_FROM_OTF( py_visits, NPY_INT32, NPY_ARRAY_OUT_ARRAY );
  PyObject *py_logdos = PyObject_GetAttrString( py_hist, "logdos" );
  PyObject *py_logdos_npy = PyArray_FROM_OTF( py_logdos, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY );
  PyObject *py_known_struct = PyObject_GetAttrString( py_hist, "known_state" );
  PyObject *py_known_struct_npy = PyArray_FROM_OTF( py_known_struct, NPY_UINT8, NPY_ARRAY_OUT_ARRAY );

  for ( unsigned int i=0;i<hist.size();i++ )
  {
    double *ptr = static_cast<double*>( PyArray_GETPTR1( py_visits_npy, i ) );
    *ptr = hist[i];
    ptr = static_cast<double*>( PyArray_GETPTR1( py_logdos_npy,i) );
    *ptr = logdos[i];
    uint8_t* uptr = static_cast<uint8_t*>( PyArray_GETPTR1(py_known_struct_npy,i) );
    if ( known_structures[i] ) *uptr = 1;
    else *uptr = 0;
  }
  Py_DECREF(py_visits);
  Py_DECREF(py_logdos);
  Py_DECREF(py_known_struct);
}

void Histogram::reset()
{
  for ( unsigned int i=0;i<hist.size();i++ )
  {
    hist[i] = 0;
  }
}

void Histogram::init_from_pyhist( PyObject *py_hist )
{
  import_array();
  PyObject *py_logdos = PyObject_GetAttrString( py_hist, "logdos" );
  PyObject *py_logdos_npy = PyArray_FROM_OTF( py_logdos, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
  npy_intp *dims = PyArray_DIMS( py_logdos_npy );
  logdos.resize( dims[0] );
  for ( unsigned int i=0;i<dims[0];i++ )
  {
    double* ptr = static_cast<double*>( PyArray_GETPTR1( py_logdos_npy, i ) );
    logdos[i] = *ptr;
  }
  Py_DECREF( py_logdos );
}
