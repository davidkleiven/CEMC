#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>
#include "matrix.hpp"
#include "cf_history_tracker.hpp"
#include "mc_observers.hpp"
#include <array>
#include <Python.h>

typedef std::vector< std::vector<std::string> > name_list;
typedef std::vector< std::vector< std::vector<std::vector<int> > > > cluster_list;
typedef std::vector< std::map<std::string,double> > bf_list;
typedef std::map<std::string,double> cf;

enum class Status_t {
  READY, INIT_FAILED,NOT_INITIALIZED
};

typedef std::map<std::string,std::vector<int> > tracker_t;

class CEUpdater
{
public:
  CEUpdater();
  ~CEUpdater();

  /** New copy. NOTE: the pointer has to be deleted */
  CEUpdater* copy() const;

  /** Initialize the object */
  void init( PyObject *BC, PyObject *corrFunc, PyObject *ecis, PyObject *permutations );

  /** Change values of ecis */
  void set_ecis( PyObject *ecis );

  /** Returns True if the initialization process was successfull */
  bool ok() const { return status == Status_t::READY; };

  /** Computes the energy based on the ECIs and the correlations functions */
  double get_energy();

  /** Updates the CF */
  void update_cf( PyObject *single_change );
  void update_cf( SymbolChange &single_change );

  /** Computes the spin product for one element */
  double spin_product_one_atom( unsigned int ref_indx, const std::vector< std::vector<int> > &indx_list, const std::vector<int> &dec, const std::vector<std::string> &symbs );

  /**
  Calculates the new energy given a set of system changes
  the system changes is assumed to be a python-list of tuples of the form
  [(indx1,old_symb1,new_symb1),(indx2,old_symb2,new_symb2)...]
  */
  double calculate( PyObject *system_changes );
  double calculate( std::array<SymbolChange,2> &system_changes );

  /** Resets all changes */
  void undo_changes();

  /** Clears the history */
  void clear_history();

  /** Populates the given vector with all the cluster names */
  void flattened_cluster_names( std::vector<std::string> &flattened );

  /** Returns the correlaation functions as a dictionary. Only the ones that corresponds to one of the ECIs */
  PyObject* get_cf();

  /** Returns the CF history tracker */
  const CFHistoryTracker& get_history() const{ return *history; };

  /** Read-only reference to the symbols */
  const std::vector<std::string>& get_symbols() const { return symbols; };

  /** Sets the symbols */
  void set_symbols( const std::vector<std::string> &new_symbs );

  /** CE updater should keep track of where the atoms are */
  void set_atom_position_tracker( tracker_t *new_tracker ){ tracker=new_tracker; };
private:
  void create_ctype_lookup();
  void create_permutations( PyObject *pypermutations );

  std::vector<std::string> symbols;
  name_list cluster_names;
  cluster_list cluster_indx;
  bf_list basis_functions;
  Status_t status{Status_t::NOT_INITIALIZED};
  Matrix<int> trans_matrix;
  std::map<std::string,int> ctype_lookup;
  std::map<std::string,double> ecis;
  CFHistoryTracker *history{nullptr};
  std::map< int, std::vector< std::vector<int> > > permutations;
  PyObject *atoms{nullptr};
  std::vector<MCObserver*> observers; // TODO: Not used at the moment. The accept/rejection is done in the Python code
  tracker_t *tracker{nullptr}; // Do not own this pointer

  /** Undos the latest changes keeping the tracker CE tracker updated */
  void undo_changes_tracker();

  /** Converts a system change encoded as a python tuple to a SymbolChange object */
  SymbolChange& py_tuple_to_symbol_change( PyObject *single_change, SymbolChange &symb_change );

  /** Extracts the decoration number from cluster names */
  int get_decoration_number( const std::string &cluster_name ) const;

  /** Returns true if all decoration numbers are equal */
  bool all_decoration_nums_equal( const std::vector<int> &dec_num ) const;
};
#endif
