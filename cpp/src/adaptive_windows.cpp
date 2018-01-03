#include "adaptive_windows.hpp"
#include "wang_landau_sampler.hpp"
#include "additional_tools.hpp"
#include <iostream>

using namespace std;

AdaptiveWindowHistogram::AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_width, \
 WangLandauSampler &sampler ): \
Histogram(Nbins,Emin,Emax), overall_Emin(Emin),overall_Emax(Emax),minimum_width(minimum_width), \
current_upper_bin(Nbins),sampler(&sampler){};

AdaptiveWindowHistogram::AdaptiveWindowHistogram( const Histogram &other, unsigned int minimum_width, WangLandauSampler &sampler ): Histogram(other),\
minimum_width(minimum_width), sampler(&sampler)
{
  overall_Emin = Emin;
  overall_Emax = Emax;
  current_upper_bin = Nbins;
}

AdaptiveWindowHistogram::~AdaptiveWindowHistogram()
{
  clear_updater_states();
}

bool AdaptiveWindowHistogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int count = 0;
  bool part_of_histogram_has_converged = false;
  unsigned int first_non_converged_bin = 0;
  bool converged = false;

  // If the window is small and there are a few bins with no known structure
  // The number of converged bins might never reach the minimum window size
  // Therefor calculate the maximum number of converged bins
  unsigned int maximum_possible_number_of_converged_bins = 0;
  unsigned int min_bin = 0;
  for ( unsigned int i=0;i<current_upper_bin;i++ )
  {
    if ( known_structures[i] )
    {
      maximum_possible_number_of_converged_bins += 1;
    }
  }

  // Identify the largest window from the upper energy range that is locally converged
  for ( unsigned int i=current_upper_bin;i>0;i-- )
  {
    if ( !known_structures[i-1] ) continue;

    mean += hist[i-1];
    count += 1;
    if ( hist[i-1] < minimum )
    {
      minimum = hist[i-1];
      min_bin = i-1;
    }

    double mean_dbl = static_cast<double>(mean)/count;

    // Check if the window contains the minimum number of bins to check for local convergence
    if ( (count > minimum_width) || (count >= maximum_possible_number_of_converged_bins) )
    {
      if ( minimum > criteria*mean_dbl ) part_of_histogram_has_converged=true;
    }

    // Check if the next bins makes window "non-converged". If so abort and use
    // this bin as an upper limit of the energy range
    if ( part_of_histogram_has_converged && (minimum < criteria*mean_dbl) )
    {
      first_non_converged_bin = i-1;
      break;
    }
    else if ( minimum < criteria*mean_dbl )
    {
      break;
    }
  }
  //cout << min_bin << " " << current_upper_bin << " " << minimum << " " << criteria*static_cast<double>(mean)/count << endl;

  // If first_non_converged_bin is 0 then the entire DOS has converged for this value of the modification factor
  converged = (first_non_converged_bin == 0) && part_of_histogram_has_converged;
  if ( converged )
  {
    return true;
  }

  // Update the histogram limits
  if ( part_of_histogram_has_converged )
  {
    // Check if the propsed window is valid (contains enough known stats and is large enough)
    if ( !is_valid_window(first_non_converged_bin) )
    {
      // If not valid window: return and continue sampling in the same window
      return false;
    }

    window_edges.push_back( first_non_converged_bin+1 );
    logdos_on_edges.push_back( logdos[first_non_converged_bin+1] );

    // Check if current bin equals Nbins. If so this is the first time
    // Save the sampler state at this run.
    // When algorithm restarts at a new modification factor, it will start from these states
    if ( current_upper_bin == Nbins )
    {
      get_updater_states(); // Store the current state. Will restart from this state later
    }

    current_upper_bin = first_non_converged_bin+2;
    double emin = Emin;
    double emax = get_energy( current_upper_bin );
    sampler->run_until_valid_energy( emin, emax );
    cout << "Current upper bin " << current_upper_bin << endl;
  }
  return false;
}

void AdaptiveWindowHistogram::reset()
{
  Histogram::reset();
  current_upper_bin = Nbins;
  make_dos_continous();
  window_edges.clear();
  logdos_on_edges.clear();
  sampler->set_updaters( first_change_state, atom_pos_track_first_change_state, current_bin_first_change_state ); // Distrubute the CE updaters across the energy space
}

void AdaptiveWindowHistogram::make_dos_continous()
{
  // Loop over all sub intervals and make the DOS continous at the connections
  for ( unsigned int i=0;i<window_edges.size();i++ )
  {
    double diff = logdos[window_edges[i]] - logdos_on_edges[i];

    for ( unsigned int j=0;j<window_edges[i]+1;j++ )
    {
      logdos[j] -= diff;
    }

    for ( unsigned int j=i+1;j<logdos_on_edges.size();j++ )
    {
      logdos_on_edges[j] -= diff;
    }
  }
}

bool AdaptiveWindowHistogram::bin_in_range( int bin ) const
{
  return (bin>=0) && (bin<current_upper_bin);
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

  atom_pos_track_first_change_state = sampler->get_atom_pos_trackers();
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
  if ( n_known_states <= 2 ) return false;

  return true;
}

void AdaptiveWindowHistogram::update( unsigned int bin, double modfactor )
{
  Histogram::update(bin,modfactor);
}
