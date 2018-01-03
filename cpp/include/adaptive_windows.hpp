#ifndef ADAPTIVE_WINDOW_HISTOGRAM_H
#define ADAPTIVE_WINDOW_HISTOGRAM_H
#include "histogram.hpp"
#include <vector>
#include <map>
#include <string>
//#define DEBUG_ADAPTIVE_WINDOWS

typedef std::vector< std::map< std::string,std::vector<int> > > listdict;
class WangLandauSampler;
class CEUpdater;

class AdaptiveWindowHistogram: public Histogram
{
public:
  AdaptiveWindowHistogram( unsigned int Nbins, double Emin, double Emax, unsigned int minimum_bins, WangLandauSampler &sampler );
  AdaptiveWindowHistogram( const Histogram &other, unsigned int minimum_bins, WangLandauSampler &sampler );
  virtual ~AdaptiveWindowHistogram();

  /** Returns true if the histogram is flat */
  virtual bool is_flat( double criteria ) override;

  /** Reset the histogram */
  virtual void reset() override;

  /** Return True if the bin is in range */
  virtual bool bin_in_range( int bin ) const override;

  /** Updates the historam */
  virtual void update( unsigned int bin, double modfactor ) override;

  /** Shifts the DOS to be continous at the window edges. TODO: Is not completely correct implemented */
  void make_dos_continous();

  /** Clear all the updater states */
  void clear_updater_states();
private:
  /** Stores a copy of the updater states of the Wang Landau Sampler */
  void get_updater_states();

  /** Checks that the proposed window is valid. Number of bins larger than minimum_width and there is at least two known states in the window */
  bool is_valid_window( unsigned int upper_bin ) const;

  double overall_Emin;
  double overall_Emax;
  unsigned int minimum_width;
  unsigned int current_upper_bin;
  WangLandauSampler *sampler{nullptr};
  std::vector<unsigned int> window_edges;
  std::vector<double> logdos_on_edges;
  std::vector<CEUpdater*> first_change_state;
  listdict atom_pos_track_first_change_state;
  std::vector<int> current_bin_first_change_state;
};

#endif
