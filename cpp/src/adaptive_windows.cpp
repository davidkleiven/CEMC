#include "adaptive_windows.hpp"
#include "wang_landau_sampler.hpp"
#include <iostream>

using namespace std;

AdaptiveWindowHistogram::AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_width, \
  unsigned int n_overlap_bins, WangLandauSampler &sampler ): \
Histogram(Nbins,Emin,Emax), overall_Emin(Emin),overall_Emax(Emax),minimum_width(minimum_width),n_overlap_bins(n_overlap_bins), \
current_upper_bin(Nbins),sampler(&sampler)
{
  n_overlap_bins = 1; // Has to be one when no overlap is considered
};

bool AdaptiveWindowHistogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int count = 0;
  bool part_of_histogram_has_converged = false;
  unsigned int first_non_converged_bin = 0;
  for ( unsigned int i=current_upper_bin;i>0;i-- )
  {
    if ( !known_structures[i-1] ) continue;

    mean += hist[i-1];
    count += 1;
    if ( hist[i-1] < minimum ) minimum = hist[i-1];

    double mean_dbl = static_cast<double>(mean)/count;
    if ( count > minimum_width )
    {
      if ( minimum > criteria*mean_dbl ) part_of_histogram_has_converged=true;
    }

    if ( part_of_histogram_has_converged && minimum < criteria*mean_dbl )
    {
      first_non_converged_bin = i-1;
      break;
    }
  }

  bool converged = (first_non_converged_bin == 0) && part_of_histogram_has_converged;
  if ( converged ) return true;

  // Update the histogram limits
  if ( part_of_histogram_has_converged )
  {
    window_edges.push_back( first_non_converged_bin );
    current_upper_bin = first_non_converged_bin;//+n_overlap_bins; // TODO: Has to include this when continous dos is better handles
    Emax = get_energy( current_upper_bin );
    sampler->run_until_valid_energy();
    cout << "Current Emax: " << Emax << endl;
  }

  return false;
}

void AdaptiveWindowHistogram::reset()
{
  Histogram::reset();
  current_upper_bin = Nbins;
  Emax = overall_Emax;
  make_dos_continous();
}

void AdaptiveWindowHistogram::make_dos_continous()
{
  for ( unsigned int i=0;i<window_edges.size();i++ )
  {
    double diff = logdos[window_edges[i]] - logdos[window_edges[i]+n_overlap_bins];
    unsigned int start = 0;
    if ( i == 0 ) start = 0;
    else start = window_edges[i-1];

    for ( unsigned int j=start;j<window_edges[i];i++ )
    {
      logdos[j] -= diff;
    }
  }
}
