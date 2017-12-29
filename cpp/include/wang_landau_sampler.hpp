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

class WangLandauSampler
{
public:
  WangLandauSampler( PyObject *BC, PyObject *corrFunc, PyObject *ecis, PyObject *permutations, PyObject *py_wl );
  ~WangLandauSampler();

  /** Returns a trial move that preserves the composition */
  void get_canonical_trial_move( std::array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 );
  void get_canonical_trial_move( unsigned int thread_num, std::array<SymbolChange,2> &changes, unsigned int &select1, unsigned int &select2 );

  /** Perform random steps until the position is inside the energy range */
  void run_until_valid_energy();

  /** Running one MC step */
  void step();

  /** Run the WL sampler */
  void run( unsigned int nsteps );

  /** Sends all results back to the Python object */
  void send_results_to_python();
private:
  std::vector<CEUpdater*> updaters; // Keep one updater for each thread
  std::vector< std::map< std::string,std::vector<int> > > atom_positions_track;
  bool ready{true};
  double f{2.71};
  double min_f{1E-6};
  double flatness_criteria{0.8};
  unsigned int check_convergence_every{1000};
  std::vector<int> site_types;
  std::vector<std::string> symbols;
  Histogram *histogram{nullptr};
  double current_energy{0.0};
  //unsigned int current_bin;
  std::vector<int> current_bin;
  PyObject *py_wl{nullptr};
  bool converged{false};
  std::vector<unsigned int> seeds;
  unsigned int iter{0};
};
#endif
