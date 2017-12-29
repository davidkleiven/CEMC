#ifndef ADAPTIVE_WINDOW_HISTOGRAM_H
#define ADAPTIVE_WINDOW_HISTOGRAM_H
#include "histogram.hpp"
#include <vector>

class WangLandauSampler;

class AdaptiveWindowHistogram: public Histogram
{
public:
  AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_bins, unsigned int n_overlap_bins, WangLandauSampler &sampler );

  /** Returns true if the histogram is flat */
  virtual bool is_flat( double criteria );

  /** Reset the histogram */
  virtual void reset();

  /** Shifts the DOS to be continous at the window edges. TODO: Is not completely correct implemented */
  void make_dos_continous();
private:
  double overall_Emin;
  double overall_Emax;
  unsigned int minimum_width;
  unsigned int n_overlap_bins;
  unsigned int current_uppder_bin;
  WangLandauSampler *sampler{nullptr};
  std::vector<unsigned int> window_edges;
};

#endif
