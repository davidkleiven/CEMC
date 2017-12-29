#ifndef ADAPTIVE_WINDOW_HISTOGRAM_H
#define ADAPTIVE_WINDOW_HISTOGRAM_H
#include "histogram.hpp"

class AdaptiveWindowHistogram: public Histogram
{
public:
  AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_bins, unsigned int n_overlap_bins );

  /** Returns true if the histogram is flat */
  virtual bool is_flat( double criteria );

  /** Reset the histogram */
  virtual void reset();
private:
  double overall_Emin;
  double overall_Emax;
  unsigned int minimum_width;
  unsigned int n_overlap_bins;
  unsigned int current_uppder_bin;
};

#endif
