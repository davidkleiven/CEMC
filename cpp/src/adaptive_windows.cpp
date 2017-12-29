#include "adaptive_windows.hpp"

AdaptiveWindowHistogram::AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_width, \
  unsigned int n_overlap_bins ):
Histogram(Nbins,Emin,Emax), overall_Emin(Emin),overall_Emax(Emax),minimum_width(minimum_width),n_overlap_bins(n_overlap_bins), \
current_uppder_bin(Nbins){};

bool AdaptiveWindowHistogram::is_flat( double criteria )
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  unsigned int count = 0;
  bool part_of_histogram_has_converged = false;
  unsigned int first_non_converged_bin = 0;
  for ( unsigned int i=current_uppder_bin;i>0;i-- )
  {
    if ( !known_structures[i-1] ) continue;

    mean += hist[i-1];
    count += 1;
    if ( hist[i-1] < minimum ) minimum = hist[i-1];

    double mean_dbl = static_cast<double>(mean)/count;
    if ( count > minimum_width )
    {
      if ( minimum > criteria*mean ) part_of_histogram_has_converged=true;
    }

    if ( part_of_histogram_has_converged && minimum < criteria*mean )
    {
      first_non_converged_bin = i-1;
      break;
    }
  }

  // Update the histogram limits
  if ( part_of_histogram_has_converged )
  {
    current_uppder_bin = first_non_converged_bin+n_overlap_bins;
    Emax = get_energy( current_uppder_bin );
  }
  bool converged = (first_non_converged_bin == 0) && part_of_histogram_has_converged;
  return converged;
}

void AdaptiveWindowHistogram::reset()
{
  Histogram::reset();
  current_uppder_bin = Nbins;
  Emax = overall_Emax;
}
