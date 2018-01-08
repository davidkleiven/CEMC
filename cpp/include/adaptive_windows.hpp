#ifndef ADAPTIVE_WINDOW_HISTOGRAM_H
#define ADAPTIVE_WINDOW_HISTOGRAM_H
#include "histogram.hpp"
#include <vector>
#include <map>
#include <string>
//#define DEBUG_ADAPTIVE_WINDOWS

class WangLandauSampler;
class CEUpdater;
typedef std::vector< std::map< std::string,std::vector<int> > > listdict;
struct CEState
{
  CEUpdater *updater{nullptr};
  int bin;
  std::map< std::string,std::vector<int> > atom_pos_track;
};

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

  /** Distributes the random walkers evenly in the window */
  void distribute_random_walkers_evenly();

  /** Print a status message */
  void status_report( double mean, double minimum, double maximum, double criteria );

  /** Redistributes the samplers */
  virtual void redistribute_samplers() override;

  /** Returns True if the window is so small that the probability of data conflict is too large */
  virtual bool update_synchronized( unsigned int num_threads, double conflict_prob ) const;
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
  std::vector<CEState> states;
  bool has_ce_states{false};

  bool minimum_ok( double mean, double criteria, double minimum ) const;
  bool maximum_ok( double mean, double criteria, double maximum ) const;
  clock_t last_stat_report{0};
  unsigned int stat_report_every{60};
};

#endif
