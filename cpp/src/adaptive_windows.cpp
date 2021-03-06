#include "adaptive_windows.hpp"
#include "wang_landau_sampler.hpp"
#include "additional_tools.hpp"
#include <iostream>
#include <cassert>
#include <omp.h>
#include <ctime>

using namespace std;

AdaptiveWindowHistogram::AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_width, \
 WangLandauSampler &sampler ): \
Histogram(Nbins,Emin,Emax), overall_Emin(Emin),overall_Emax(Emax),minimum_width(minimum_width), \
current_upper_bin(Nbins),sampler(&sampler)
{
  states.resize(Nbins);
};

AdaptiveWindowHistogram::AdaptiveWindowHistogram( const Histogram &other, unsigned int minimum_width, WangLandauSampler &sampler ): Histogram(other),\
minimum_width(minimum_width), sampler(&sampler)
{
  overall_Emin = Emin;
  overall_Emax = Emax;
  current_upper_bin = Nbins;
  states.resize(Nbins);
}

AdaptiveWindowHistogram::~AdaptiveWindowHistogram()
{
  clear_updater_states();

  for ( unsigned int i=0;i<states.size();i++ )
  {
    delete states[i].updater;
  }
}

bool AdaptiveWindowHistogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int maximum = 0;
  unsigned int count = 0;
  bool part_of_histogram_has_converged = false;
  unsigned int first_non_converged_bin = 0;
  bool converged = false;

  // If the window is small and there are a few bins with no known structure
  // The number of converged bins might never reach the minimum window size
  // Therefor calculate the maximum number of converged bins
  unsigned int maximum_possible_number_of_converged_bins = 0;

  for ( unsigned int i=0;i<current_upper_bin-n_overlap;i++ )
  {
    if ( known_structures[i] )
    {
      maximum_possible_number_of_converged_bins += 1;
    }
  }
  bool min_ok = false;
  double mean_dbl = 0.0;

  // Identify the largest window from the upper energy range that is locally converged
  for ( unsigned int i=current_upper_bin-n_overlap;i>0;i-- )
  {
    if ( !known_structures[i-1] ) continue;

    mean += hist[i-1];
    count += 1;
    if ( hist[i-1] < minimum )
    {
      minimum = hist[i-1];
      current_min_bin = i-1;
    }

    if ( hist[i-1] > maximum )
    {
      maximum = hist[i-1];
      current_max_bin = i-1;
    }

    mean_dbl = static_cast<double>(mean)/count;
    min_ok =  minimum_ok(mean_dbl,criteria,minimum);

    // Check if the window contains the minimum number of bins to check for local convergence
    if ( (count > minimum_width) || (count >= maximum_possible_number_of_converged_bins) )
    {
      if ( min_ok ) part_of_histogram_has_converged=true;
    }

    // Check if the next bins makes window "non-converged". If so abort and use
    // this bin as an upper limit of the energy range
    if ( part_of_histogram_has_converged && !min_ok )
    {
      first_non_converged_bin = i-1;
      break;
    }
    else if ( !min_ok )
    {
      first_non_converged_bin = i-1;
      break;
    }
  }

  if ( (clock()-last_stat_report)/CLOCKS_PER_SEC > stat_report_every )
  {
    status_report( mean_dbl, minimum, maximum, criteria, current_upper_bin-first_non_converged_bin-1 );
    cout << "Min bin: " << current_min_bin << " Max bin: " << current_max_bin << endl;
    //cout << hist << endl;
  }

  // If first_non_converged_bin is 0 then the entire DOS has converged for this value of the modification factor
  converged = (first_non_converged_bin == 0) && min_ok;
  if ( converged )
  {
    return true;
  }

  // Update the histogram limits
  if ( part_of_histogram_has_converged )
  {
    // If the proposed winow is not valid, expand it until it is valid
    // Arbitrary upper boundary of current_upper_bin-6 (could be anything smaller than current_upper_bin-1)
    if( !is_valid_window(first_non_converged_bin) )
    {
      return false;
    }

    window_edges.push_back( first_non_converged_bin+1 );
    vector<double> new_overlap;
    for ( unsigned int n=0;n<n_overlap;n++ )
    {
      new_overlap.push_back(logdos[first_non_converged_bin+1+n]);
    }
    logdos_on_edges.push_back( new_overlap );

    // Check if current bin equals Nbins. If so this is the first time
    // Save the sampler state at this run.
    // When algorithm restarts at a new modification factor, it will start from these states
    if ( current_upper_bin == Nbins )
    {
      get_updater_states(); // Store the current state. Will restart from this state later
      has_ce_states = true;
    }

    current_upper_bin = first_non_converged_bin+n_overlap+1;
    double emin = Emin;
    double emax = get_energy( current_upper_bin );
    distribute_random_walkers_evenly();
    sampler->run_until_valid_energy( emin, emax );
    cout << "Current upper bin " << current_upper_bin << endl;
  }
  //cout << hist[0] << " " << hist[1] << " " << logdos[0] << " " << logdos[1] << " " << minimum << " " << current_upper_bin << endl;
  return false;
}

void AdaptiveWindowHistogram::reset()
{
  Histogram::reset();
  current_upper_bin = Nbins;
  make_dos_continous();
  window_edges.clear();
  logdos_on_edges.clear();

  if ( has_ce_states )
  {
    sampler->set_updaters( first_change_state, atom_pos_track_first_change_state, current_bin_first_change_state ); // Distrubute the CE updaters across the energy space
  }
}

void AdaptiveWindowHistogram::make_dos_continous()
{
  // Loop over all sub intervals and make the DOS continous at the connections
  for ( unsigned int i=0;i<window_edges.size();i++ )
  {
    // Reset the overlap bins except the first
    for ( unsigned int j=1;j<logdos_on_edges[i].size();j++ )
    {
      logdos[window_edges[i]+j] = logdos_on_edges[i][j];
    }

    double diff = logdos[window_edges[i]] - logdos_on_edges[i][0];

    for ( unsigned int j=0;j<window_edges[i]+1;j++ )
    {
      logdos[j] -= diff;
    }

    for ( unsigned int j=i+1;j<logdos_on_edges.size();j++ )
    {
      for ( unsigned int k=0;k<logdos_on_edges[j].size();k++ )
      {
        logdos_on_edges[j][k] -= diff;
      }
    }
  }
}

bool AdaptiveWindowHistogram::bin_in_range( int bin ) const
{
  return (bin>=0) && (bin<static_cast<int>(current_upper_bin));
}

void AdaptiveWindowHistogram::clear_updater_states()
{
  for ( unsigned int i=0;i<first_change_state.size();i++ )
  {
    delete first_change_state[i];
  }
  first_change_state.clear();
}

void AdaptiveWindowHistogram::get_updater_states()
{
  clear_updater_states();
  unsigned int n_updaters = sampler->get_n_updaters();
  for ( unsigned int i=0;i<n_updaters;i++ )
  {
    first_change_state.push_back( sampler->get_updater(i)->copy() );
  }

  atom_pos_track_first_change_state.clear();
  list_dictptr cur_atom_pos = sampler->get_atom_pos_trackers();
  for ( unsigned int i=0;i<cur_atom_pos.size();i++ )
  {
    atom_pos_track_first_change_state.push_back(*cur_atom_pos[i]);
  }
  current_bin_first_change_state = sampler->get_current_bins();
}

bool AdaptiveWindowHistogram::is_valid_window( unsigned int upper_bin ) const
{
  if ( upper_bin <= minimum_width ) return false;

  unsigned int n_known_states = 0;
  for ( unsigned int i=0;i<upper_bin;i++ )
  {
    if ( known_structures[i] ) n_known_states += 1;
  }
  //cout << "Number of known states in window: " << n_known_states << endl;
  if ( n_known_states <= minimum_width/2 ) return false;

  return true;
}

void AdaptiveWindowHistogram::update( unsigned int bin, double modfactor )
{
  if ( states[bin].updater == nullptr )
  {
    unsigned int uid = omp_get_thread_num();
    #pragma omp critical(track_new_state)
    {
      // Check this one more time inside a critical section, to be absolutely sure
      // that two processors don't allocate the same memory
      if ( states[bin].updater == nullptr )
      {
        states[bin].updater = sampler->get_updater(uid)->copy();
        states[bin].bin = bin;
        states[bin].atom_pos_track = sampler->get_atom_pos_tracker(uid);
      }
    }
  }

  Histogram::update(bin,modfactor);
}

void AdaptiveWindowHistogram::distribute_random_walkers_evenly()
{
  // Make sure that there are known states in the window
  unsigned int n_known = 0;
  for ( unsigned int i=0;i<current_upper_bin;i++ )
  {
    if ( states[i].updater != nullptr ) n_known += 1;
  }

  if ( n_known <= sampler->num_threads )
  {
    return;
  }

  unsigned int n_threads = sampler->num_threads;
  double delta = current_upper_bin/static_cast<double>(n_threads+1);

  listdict atom_track;
  std::vector<int> current_bins;
  std::vector<CEUpdater*> updaters;

  for ( unsigned int i=0;i<n_threads;i++ )
  {
    unsigned int indx = (i+1)*delta;
    unsigned int shift_pos=2*current_upper_bin;
    for ( unsigned int j=indx;j<current_upper_bin;j++ )
    {
      if ( states[j].updater != nullptr )
      {
        shift_pos = j-indx;
        break;
      }
    }

    unsigned int shift_neg=2*current_upper_bin;
    for ( int j=indx;j>=0;j-- )
    {
      if ( states[j].updater != nullptr )
      {
        shift_neg = indx-j;
        break;
      }
    }
    unsigned int new_bin = 0;
    if ( shift_pos <= shift_neg )
    {
      new_bin = indx+shift_pos;
    }
    else
    {
      new_bin = indx-shift_neg;
      //new_bin = indx+shift_pos;
    }

    updaters.push_back( states[new_bin].updater );
    current_bins.push_back( states[new_bin].bin );
    atom_track.push_back( states[new_bin].atom_pos_track );
  }
  sampler->set_updaters( updaters, atom_track, current_bins );
  cout << current_bins << endl;
}

void AdaptiveWindowHistogram::redistribute_samplers()
{
  cout << "Redistributing the samplers\n";
  distribute_random_walkers_evenly();
}

bool AdaptiveWindowHistogram::minimum_ok( double mean, double criteria, double minimum ) const
{
  return minimum > criteria*mean;
}

bool AdaptiveWindowHistogram::maximum_ok( double mean, double criteria, double maximum ) const
{
  return maximum < mean/criteria;
}

void AdaptiveWindowHistogram::status_report( double mean, double minimum, double maximum, double criteria, unsigned int converged_window )
{
  cout << "Mean: " << mean << " Min: " << minimum << " Min crit. " << mean*criteria << " Max: " << maximum << " Max crit.: " << mean/criteria;
  cout << " Converged: " << converged_window << endl;
  last_stat_report = clock();
}

bool AdaptiveWindowHistogram::update_synchronized( unsigned int num_threads, double conflict_prob ) const
{
  return static_cast<double>(num_threads)/current_upper_bin > conflict_prob;
}
