#ifndef WANG_LANDAU_SAMPLER_H
#define WAND_LANDAU_SAMPLER_H
#include <vector>
#include "ce_updater.hpp"
#include <Python.h>
#include <map>
#include <string>

class WangLandauSampler
{
public:
  WangLandauSampler( const CEUpdater &updater, PyObject *py_wl );
  ~WangLandauSampler();
private:
  std::vector<CEUpdater*> updaters; // Keep one updater for each thread
  std::map< std::string,std::vector<int> > atom_positions_track;
  bool ready{true};
  double f{2.71};
  double min_f{1E-6};
  double flatness_criteria{0.8};
  unsigned int check_convergence_every{1000};
  std::vector<int> site_types;
  std::vector<std::string> symbols;
};
#endif
