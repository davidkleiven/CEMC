#ifndef WANG_LANDAU_SAMPLER_H
#define WAND_LANDAU_SAMPLER_H
#include <vector>
#include "ce_updater.hpp"
#include "cf_history_tracker.hpp"
#include "histogram.hpp"
#include <Python.h>
#include <map>
#include <string>
#include <array>
#define WANG_LANDAU_DEBUG

typedef std::vector< std::map< std::string,std::vector<int> >* > list_dictptr;
typedef std::vector< std::map< std::string,std::vector<int> > > listdict;

class WangLandauSampler
{
public:
  WangLandauSampler( PyObject *BC, PyObject *corrFunc, PyObject *ecis, PyObject *py_wl );
  ~WangLandauSampler();

  /** Returns a trial move that preserves the composition */
  void get_canonical_trial_move( std::array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 );
  void get_canonical_trial_move( unsigned int thread_num, std::array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 );

  /** Perform random steps until the position is inside the energy range */
  void run_until_valid_energy( double emin, double emax );

  /** Running one MC step */
  void step();

  /** Run the WL sampler */
  void run( unsigned int nsteps );

  /** Sends all results back to the Python object */
  void send_results_to_python();

  /** Use a histogram with adaptive windows */
  void use_adaptive_windows( unsigned int minimum_window_width );

  /** Return the number of CE updaters */
  unsigned int get_n_updaters() const { return updaters.size(); };

  /** Get a pointer to a CE updater */
  CEUpdater* get_updater( unsigned int indx ){ return updaters[indx]; };

  /** Deletes all the CE updaters */
  void delete_updaters();

  /** Replaces the CE updaters with the onesin new_uptaders */
  void set_updaters( const std::vector<CEUpdater*> &new_updaters, listdict &new_pos_track, std::vector<int> &curr_bin );

  /** Returns a copy of the atom position trackers */
  list_dictptr get_atom_pos_trackers() const { return atom_positions_track; };

  std::map<std::string, std::vector<int> > get_atom_pos_tracker( unsigned int uid ) const { return *atom_positions_track[uid]; };

  /** Returns a copy of current bins */
  std::vector<int> get_current_bins() const { return current_bin; };

  int get_current_bin( unsigned int uid ) const { return current_bin[uid]; };

  /** Get Monte Carlo time (n_iter/n_bins) */
  double get_mc_time() const;

  bool use_inverse_time_algorithm{false};

  static const unsigned int num_threads;

  void save_sub_bin_distribution( const std::string &fname ){ histogram->save_sub_bin_distribution(fname); };

  /** Save convergence time */
  void save_convergence_time( const std::string &fname ) const;
private:
  /** Updates the atom position trackers */
  void update_atom_position_track( unsigned int uid, std::array<SymbolChange,2> &changes, unsigned int select1, unsigned int select2 );

  /** Upates all bins above the current bin */
  void update_all_above();

  /** Updates the current bin */
  void update_current();

  std::vector<CEUpdater*> updaters; // Keep one updater for each thread
  list_dictptr atom_positions_track;
  bool ready{true};
  double f{2.71};
  double min_f{1E-6};
  double flatness_criteria{0.8};
  unsigned int check_convergence_every{1000};
  std::vector<int> site_types;
  std::vector<std::string> symbols;
  Histogram *histogram{nullptr};
  //double current_energy{0.0};
  std::vector<int> current_bin;
  PyObject *py_wl{nullptr};
  bool converged{false};
  std::vector<unsigned int> seeds;
  double iter{0}; // Store as double to avoid overflow
  bool inverse_time_activated{false};
  double iter_since_last{0};
  double n_outside_range{0};
  double n_self_proposals{0.0};
  std::vector<double> time_to_converge;
  double inv_time_factor{1.0};
  unsigned int update_hist_every{5};
  std::vector<bool> is_first;
  std::vector<double> current_energy;
  double avg_bin_change{0.0};
  double avg_acc_rate{0.0};
};
#endif
