#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>
#include "matrix.hpp"
#include "cf_history_tracker.hpp"
#include <Python.h>

typedef std::vector< std::vector<std::string> > name_list;
typedef std::vector< std::vector< std::vector<std::vector<int> > > > cluster_list;
typedef std::vector< std::map<std::string,double> > bf_list;
typedef std::map<std::string,double> cf;

enum class Status_t {
  READY, INIT_FAILED,NOT_INITIALIZED
};

class CEUpdater
{
public:
  CEUpdater();

  /** Initialize the object */
  void init( PyObject *BC, PyObject *corrFunc, PyObject *ecis );

  /** Returns True if the initialization process was successfull */
  bool ok() const { return status == Status_t::READY; };

  /** Computes the energy based on the ECIs and the correlations functions */
  double get_energy();

  /** Updates the CF */
  void update_cf( PyObject *single_change );

  /** Computes the spin product for one element */
  double spin_product_one_atom( unsigned int ref_indx, const std::vector< std::vector<int> > &indx_list, const std::vector<int> &dec );

  /** Calculates the new energy given a set of system changes
  the system changes is assumed to be a python-list of tuples of the form
  [(indx1,old_symb1,new_symb1),(indx2,old_symb2,new_symb2)...]
  */
  double calculate( PyObject *system_changes );

  /** Resets all changes */
  void undo_changes();

  /** Clears the history */
  void clear_history();
private:
  void create_ctype_lookup();
  void create_permutations();

  std::vector<std::string> symbols;
  name_list cluster_names;
  cluster_list cluster_indx;
  bf_list basis_functions;
  Status_t status{Status_t::NOT_INITIALIZED};
  Matrix<int> trans_matrix;
  std::map<std::string,int> ctype_lookup;
  std::map<std::string,double> ecis;
  CFHistoryTracker history;
  std::map< int, std::vector< std::vector<int> > > permutations;
};
#endif
