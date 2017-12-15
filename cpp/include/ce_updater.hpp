#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>
#include "matrix.hpp"
#include "cf_history_tracker.hpp"
#include <Python.h>

typedef std::vector< std::vector<std::string> > name_list;
typedef std::vector< std::vector< std::vector<int> > > cluster_list;
typedef std::vector< std::map<std::string,double> > bf_list;
typedef std::map<std::string,double> cf;

enum class Status_t {
  READY, INIT_FAILED
};

class CEUpdater
{
public:
  CEUpdater( PyObject *BC, PyObject *corrFunc, PyObject *ecis );

  /** Returns True if the initialization process was successfull */
  bool ok() const { return status == Status_t::READY; };

  /** Computes the energy based on the ECIs and the correlations functions */
  double get_energy() const;

  /** Updates the CF */
  void update_cf( PyObject *system_chagnes );

  /** Computes the spin product for one element */
  double spin_product_one_atom( unsigned int ref_indx, const std::vector< std::vector<int> > &indx_list, const std::vector<int> &dec );
private:
  void create_ctype_lookup();

  std::vector<std::string> symbols;
  name_list cluster_names;
  cluster_list cluster_indx;
  bf_list basis_functions;
  cf* corr_functions{nullptr};
  Status_t status{Status_t::READY};
  Matrix<int> trans_matrix;
  std::map<std::string,int> ctype_lookup;
  std::map<std::string,double> ecis;
  CFHistoryTracker history;
};
#endif
