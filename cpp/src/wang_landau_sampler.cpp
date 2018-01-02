#include "wang_landau_sampler.hpp"
#include "additional_tools.hpp"
#include "adaptive_windows.hpp"
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
using namespace std;

const static unsigned int num_threads = omp_get_max_threads(); // Use the maximum number of threads

WangLandauSampler::WangLandauSampler( PyObject *BC, PyObject *corrFunc, PyObject *ecis, PyObject *permutations, PyObject *py_wl_in )
{
  // Initialize the seeds for the different threads
  srand(time(NULL));
  for ( unsigned int i=0;i<num_threads;i++ )
  {
    seeds.push_back( rand() );
  }

  CEUpdater updater;
  updater.init( BC,corrFunc,ecis,permutations);

  #ifdef WANG_LANDAU_DEBUG
    cout << "Initializing the wanglandau object\n";
  #endif
  py_wl = py_wl_in;
  for ( unsigned int i=0;i<num_threads;i++ )
  {
    updaters.push_back( updater.copy() );
  }

  #ifdef WANG_LANDAU_DEBUG
    cout << "Reading symbols from atom position track\n";
  #endif
  // Read the symbol position array from the py_object
  PyObject *py_symbpos = PyObject_GetAttrString( py_wl, "atom_positions_track" );
  if ( py_symbpos == nullptr )
  {
    ready = false;
    return;
  }

  #ifdef WANG_LANDAU_DEBUG
    cout << "Reading atom position track\n";
  #endif
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  map<string,vector<int> > temp_atom_positions_track;
  while ( PyDict_Next(py_symbpos, &pos, &key, &value) )
  {
    vector<int> pos;
    int size = PyList_Size(value);
    for ( int i=0;i<size;i++ )
    {
      pos.push_back( PyInt_AsLong(PyList_GetItem(value,i)) );
    }
    temp_atom_positions_track[PyString_AsString(key)] = pos;
  }
  Py_DECREF(py_symbpos);
  for ( unsigned int i=0;i<num_threads;i++ )
  {
    atom_positions_track.push_back(temp_atom_positions_track);
  }

  #ifdef WANG_LANDAU_DEBUG
    cout << "Reading fmin and f from Python object\n";
  #endif
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


  #ifdef WANG_LANDAU_DEBUG
    cout << "Reading site types\n";
  #endif
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

  #ifdef WANG_LANDAU_DEBUG
    cout << "Reading histogram data\n";
  #endif
  PyObject *py_hist = PyObject_GetAttrString( py_wl, "histogram" );
  PyObject *py_nbins = PyObject_GetAttrString( py_hist, "Nbins" );
  int Nbins = PyInt_AsLong( py_nbins );
  Py_DECREF( py_nbins );

  PyObject *py_emin = PyObject_GetAttrString( py_hist, "Emin" );
  double Emin = PyFloat_AS_DOUBLE(py_emin);
  Py_DECREF( py_emin );

  PyObject *py_emax = PyObject_GetAttrString( py_hist, "Emax" );
  double Emax = PyFloat_AS_DOUBLE( py_emax );
  Py_DECREF(py_emax);

  histogram = new Histogram( Nbins, Emin, Emax );
  histogram->init_from_pyhist( py_hist );
  Py_DECREF( py_hist );

  PyObject *py_cur_bin = PyObject_GetAttrString( py_wl, "current_bin" );
  unsigned int curbin = PyInt_AsLong(py_cur_bin);
  Py_DECREF(py_cur_bin);
  for ( unsigned int i=0;i<num_threads;i++ )
  {
    current_bin.push_back(curbin);
  }

  PyObject *py_iter = PyObject_GetAttrString( py_wl, "iter" );
  iter = PyInt_AsLong(py_iter);
  Py_DECREF(py_iter);
  #ifdef WANG_LANDAU_DEBUG
    cout << "Wang Landau object initialized successfully!\n";
  #endif
}

WangLandauSampler::~WangLandauSampler()
{
  for ( unsigned int i=0;i<updaters.size();i++ )
  {
    delete updaters[i];
  }
  delete histogram;
}

void WangLandauSampler::get_canonical_trial_move( array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 )
{
  get_canonical_trial_move(0,changes,select1,select2);
}

void WangLandauSampler::get_canonical_trial_move( unsigned int thread_num, array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 )
{
  unsigned int first_indx = rand_r(&seeds[thread_num])%symbols.size();
  const string& symb1 = symbols[first_indx];
  unsigned int N = atom_positions_track[thread_num][symb1].size();
  select1 = rand_r(&seeds[thread_num])%N;
  unsigned int indx1 = atom_positions_track[thread_num][symb1][select1];
  unsigned int site_type1 = site_types[indx1];

  string symb2 = symb1;
  unsigned int site_type2 = site_type1+1;
  unsigned int indx2 = indx1;
  while ( (symb2 == symb1) || (site_type2 != site_type1) )
  {
    symb2 = symbols[rand_r( &seeds[thread_num] )%symbols.size()];
    N = atom_positions_track[thread_num][symb2].size();
    select2 = rand_r( &seeds[thread_num] )%N;
    indx2 = atom_positions_track[thread_num][symb2][select2];
    site_type2 = site_types[indx2];
  }

  changes[0].indx = indx1;
  changes[0].old_symb = symb1;
  changes[0].new_symb = symb2;
  changes[1].indx = indx2;
  changes[1].old_symb = symb2;
  changes[1].new_symb = symb1;
}

void WangLandauSampler::step()
{
  iter++;
  int uid = omp_get_thread_num();
  array<SymbolChange,2> change;
  unsigned int select1, select2;
  get_canonical_trial_move( uid, change, select1, select2 );

  double energy = updaters[uid]->calculate( change );

  int bin = histogram->get_bin( energy );

  if ( !histogram->bin_in_range(bin) )
  {
    updaters[uid]->undo_changes();
    return;
  }

  double dosratio = histogram->get_dos_ratio_old_divided_by_new( current_bin[uid], bin );
  double uniform_random = static_cast<double>(rand_r( &seeds[uid] ))/RAND_MAX;

  if ( uniform_random < dosratio )
  {
    // Accept
    const string& symb1_old = change[0].old_symb;
    const string& symb2_old = change[1].old_symb;
    unsigned int indx1 = change[0].indx;
    unsigned int indx2 = change[1].indx;
    atom_positions_track[uid][symb1_old][select1] = indx2;
    atom_positions_track[uid][symb2_old][select2] = indx1;
    current_bin[uid] = bin;
  }
  else
  {
    updaters[uid]->undo_changes();
  }
  updaters[uid]->clear_history();
  histogram->update( current_bin[uid], f );
}

void WangLandauSampler::run( unsigned int nsteps )
{
  if ( check_convergence_every%num_threads != 0 )
  {
    check_convergence_every = (check_convergence_every/num_threads+1)*num_threads;
  }

  unsigned int n_outer = nsteps/check_convergence_every;
  for ( unsigned int i=0;i<n_outer;i++ )
  {
    #pragma omp parallel for
    for ( unsigned int j=0;j<check_convergence_every;j++ )
    {
      step();
    }

    if ( histogram->is_flat( flatness_criteria) )
    {
      f /= 2.0;
      histogram->reset();
      cout << "Converged! New f: " << f << endl;
    }

    if ( f < min_f )
    {
      cout << "Simulation converged!\n";
      converged = true;
      break;
    }
  }
  send_results_to_python();
}

void WangLandauSampler::send_results_to_python()
{
  PyObject *py_hist = PyObject_GetAttrString( py_wl, "histogram" );
  histogram->send_to_python_hist( py_hist );
  cout << "here_sending\n";
  Py_DECREF(py_hist);

  PyObject *cur_f = PyFloat_FromDouble(f);
  PyObject_SetAttrString( py_wl, "f", cur_f );
  Py_DECREF( cur_f );

  if ( converged )
  {
    PyObject_SetAttrString( py_wl, "converged", Py_True );
  }
  else
  {
    PyObject_SetAttrString( py_wl, "converged", Py_False );
  }

  PyObject *py_iter = PyInt_FromLong(iter);
  PyObject_SetAttrString( py_wl, "iter", py_iter );
  Py_DECREF( py_iter );
}

void WangLandauSampler::run_until_valid_energy()
{
  bool all_ok = true;
  // Check if all processors are in a valid energy state
  for ( unsigned int i=0;i<current_bin.size();i++ )
  {
    if ( !histogram->bin_in_range(current_bin[i]) )
    {
      all_ok = false;
      break;
    }
  }

  if ( all_ok )
  {
    #ifdef WANG_LANDAU_DEBUG
      cout << "All processors in valid energy range\n";
    #endif
    return;
  }
  else
  {
    // Check if one of the processors is inside the energy range
    int proc_in_valid_state = -1;
    for ( unsigned int i=0;i<current_bin.size();i++ )
    {
      if ( histogram->bin_in_range(current_bin[i]) )
      {
        proc_in_valid_state = i;
        break;
      }
    }
    if ( proc_in_valid_state != -1 )
    {
      #ifdef WANG_LANDAU_DEBUG
        cout << "At least one processor in valid energy range. Set all other samples equal to this one\n";
      #endif
      // Update the updaters
      for ( int i=0;i<current_bin.size();i++ )
      {
        if ( i == proc_in_valid_state ) continue;

        delete updaters[i];
        updaters[i] = updaters[proc_in_valid_state]->copy();
        current_bin[i] = current_bin[proc_in_valid_state];
      }
      return;
    }
  }

  #ifdef WANG_LANDAU_DEBUG
    cout << "All processors out of energy range. Generate random states to get them inside the energy range\n";
  #endif

  // All processors are outside of the valid range
  unsigned int max_steps = 1000000;
  bool found_state_in_range = false;
  for ( unsigned int i=0;i<max_steps;i++ )
  {
    array<SymbolChange,2> change;
    unsigned int select1, select2;
    get_canonical_trial_move( 0, change, select1, select2 );

    double energy = updaters[0]->calculate( change );
    int bin = histogram->get_bin( energy );

    if ( histogram->bin_in_range(bin) )
    {
      found_state_in_range = true;
      #ifdef WANG_LANDAU_DEBUG
        cout << "Found a valid state in " << i << " trial moves\n";
      #endif
      break;
    }

    double current_energy = histogram->get_energy( current_bin[0] );
    if ( current_energy > histogram->get_emax() )
    {
      if ( energy < current_energy )
      {
        updaters[0]->clear_history();
        current_bin[0] = bin;
      }
      else
      {
        updaters[0]->undo_changes();
      }
    }
    else
    {
      if ( current_energy < histogram->get_emin() )
      {
        if ( energy > current_energy )
        {
          updaters[0]->clear_history();
          current_bin[0] = bin;
        }
        else
        {
          updaters[0]->undo_changes();
        }
      }
    }
  }

  // Copy all states
  for ( unsigned int i=1;i<updaters.size();i++ )
  {
    delete updaters[i];
    updaters[i] = updaters[0]->copy();
    current_bin[i] = current_bin[0];
  }

  if ( !found_state_in_range )
  {
    throw runtime_error( "Could not find any energy state inside the histogram range!" );
  }
}

void WangLandauSampler::use_adaptive_windows( unsigned int minimum_window_width )
{
  unsigned int Nbins = histogram->get_nbins();
  double Emin = histogram->get_emin();
  double Emax = histogram->get_emax();
  delete histogram;
  histogram = new AdaptiveWindowHistogram( Nbins, Emin, Emax, minimum_window_width, *this );
}
