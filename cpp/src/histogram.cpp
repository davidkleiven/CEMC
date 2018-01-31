#include "histogram.hpp"
#include <numpy/ndarrayobject.h>
#include <iostream>
#include "additional_tools.hpp"
#include <fstream>

using namespace std;

Histogram::Histogram( unsigned int Nbins, double Emin, double Emax ):Nbins(Nbins),Emin(Emin),Emax(Emax)
{
  //import_array();
  hist.resize(Nbins);
  logdos.resize(Nbins);
  known_structures.resize(Nbins);
  bin_transfer.resize(Nbins);

  for ( unsigned int i=0;i<Nbins;i++ )
  {
    hist[i] = 0;
    logdos[i] = 0.0;
    known_structures[i] = false;
    for ( unsigned int j=0;j<bin_transfer[i].size();j++ )
    {
      bin_transfer[i][j] = 0;
    }
  }
}

Histogram::Histogram( const Histogram &other )
{
  swap(*this,other);
}

void Histogram::init_sub_bins()
{
  cout << sub_bin_distribution.size() << endl;
  for ( unsigned int i=0;i<sub_bin_distribution.size();i++ )
  {
    delete sub_bin_distribution[i];
  }
  sub_bin_distribution.clear();

  for ( unsigned int i=0;i<Nbins;i++ )
  {
    double emin = get_energy(i);
    double emax = get_energy(i+1);
    sub_bin_distribution.push_back( new Histogram(10,emin,emax) );
  }

}

Histogram::~Histogram()
{
  for ( unsigned int i=0;i<sub_bin_distribution.size();i++ )
  {
    delete sub_bin_distribution[i];
    sub_bin_distribution[i] = nullptr;
  }
  sub_bin_distribution.clear();
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
  #ifdef DEBUG_LOSS_OF_PRECISION
    double current_log_dos = logdos[bin];
    double new_log_dos = current_log_dos+mod_factor;
    if ( new_log_dos <= current_log_dos )
    {
      cerr << "Warning! Loss of presicion. Some parts of the DOS can no longer be updated\n";
    }
  #endif

  hist[bin] += 1;
  logdos[bin] += mod_factor;
}

void Histogram::update_sub_bin( unsigned int bin, double energy )
{
  if ( bin >= sub_bin_distribution.size() )
  {
    return;
  }
  int sub_bin = sub_bin_distribution[bin]->get_bin(energy);

  if ( !sub_bin_distribution[bin]->bin_in_range(sub_bin) )
  {
    return;
  }
  sub_bin_distribution[bin]->update( sub_bin, 0.1 );
}

bool Histogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int maximum = 0;
  unsigned int count = 0;
  for ( unsigned int i=0;i<hist.size();i++ )
  {
    if ( !known_structures[i] ) continue;

    mean += hist[i];
    count += 1;
    if ( hist[i] < minimum )
    {
      minimum = hist[i];
      current_min_bin = i;
    }
    else if ( hist[i] > maximum )
    {
      maximum = hist[i];
      current_max_bin = i;
    }
  }

  double mean_dbl = static_cast<double>(mean)/count;
  return minimum > criteria*mean_dbl;
}

double Histogram::get_dos_ratio_old_divided_by_new( unsigned int old_bin, unsigned int new_bin ) const
{
  double diff = logdos[old_bin] - logdos[new_bin];
  if ( diff > 0.0 ) return 1.0;
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
    int *ptr_hist = static_cast<int*>( PyArray_GETPTR1( py_visits_npy, i ) );
    *ptr_hist = hist[i];
    double* ptr = static_cast<double*>( PyArray_GETPTR1( py_logdos_npy,i) );
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
  PyObject *py_known_struct = PyObject_GetAttrString( py_hist, "known_state" );
  PyObject *py_known_struct_npy = PyArray_FROM_OTF( py_known_struct, NPY_UINT8, NPY_ARRAY_IN_ARRAY );
  npy_intp *dims = PyArray_DIMS( py_logdos_npy );
  logdos.resize( dims[0] );
  known_structures.resize( dims[0] );
  for ( unsigned int i=0;i<dims[0];i++ )
  {
    double* ptr = static_cast<double*>( PyArray_GETPTR1( py_logdos_npy, i ) );
    logdos[i] = *ptr;
    unsigned char* known_state = static_cast<unsigned char*>( PyArray_GETPTR1( py_known_struct,i) );
    known_structures[i] = *known_state;
  }
  Py_DECREF( py_logdos );
  Py_DECREF( py_known_struct );
}

void Histogram::save_sub_bin_distribution( const string &fname ) const
{
  ofstream out;
  out.open( fname.c_str() );
  if ( !out.good() )
  {
    cout << "An error occured when trying to write the sub bin distribution to file\n";
    return;
  }

  for ( unsigned int i=0;i<sub_bin_distribution.size();i++ )
  {
    for ( unsigned int j=0;j<sub_bin_distribution[i]->hist.size();j++ )
    {
      out << sub_bin_distribution[i]->hist[j] << ",";
    }
    out << "\n";
  }
  out.close();
  cout << "Sub bin distribution is written to " << fname << endl;
}

void swap( Histogram &first, const Histogram &other )
{
  first.Nbins = other.Nbins;
  first.Emin = other.Emin;
  first.Emax = other.Emax;
  first.track_states = other.track_states;

  first.hist = other.hist;
  first.logdos = other.logdos;
  first.known_structures = other.known_structures;
  first.bin_transfer = other.bin_transfer;
  first.clear_sub_hist();
  first.init_sub_bins();

}

void Histogram::clear_sub_hist()
{
  for ( unsigned int i=0;i<sub_bin_distribution.size();i++ )
  {
    delete sub_bin_distribution[i];
  }
  sub_bin_distribution.clear();
}

bool Histogram::update_synchronized( unsigned int num_threads, double conflict_prob ) const
{
  return static_cast<double>(num_threads)/Nbins > conflict_prob;
}

void Histogram::update_bin_transfer( int from, int to )
{
  int N = bin_transfer[from].size();
  int diff = to-from;
  if ( diff > N-2 )
  {
    bin_transfer[from][N-1] += 1;
  }
  else if ( diff < 1 )
  {
    bin_transfer[from][0] += 1;
  }
  else
  {
    bin_transfer[from][N/2+diff] += 1;
  }
}

void Histogram::save_bin_transfer( const string &fname ) const
{
  ofstream out;
  out.open( fname.c_str() );
  if ( !out.good() )
  {
    cout << "An error occured when opening a file to save the bin transfer\n";
    return;
  }
  for ( unsigned int i=0;i<bin_transfer.size();i++ )
  {
    for ( unsigned int j=0;j<bin_transfer[i].size();j++ )
    {
      out << bin_transfer[i][j] << ",";
    }
    out << "\n";
  }
  out.close();
}
