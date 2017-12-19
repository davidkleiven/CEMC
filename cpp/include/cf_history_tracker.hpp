#ifndef CF_HISTORY_TRACKER_H
#define CF_HISTORY_TRACKER_H
#include <string>
#include <array>
#include <Python.h>
#include <map>
#include <vector>

typedef std::map<std::string,double> cf;

struct SymbolChange
{
  int indx;
  std::string old_symb;
  std::string new_symb;
};

class CFHistoryTracker
{
public:
  CFHistoryTracker( const std::vector< std::string> &cluster_names );
  ~CFHistoryTracker();

  /** Return a pointer to the next pointer to be written to */
  void get_next( cf **next_cf, SymbolChange **symb_change );

  /** Returns a reference to the active correlation function */
  cf& get_current();

  /** Gets the system change and previus */
  void pop( SymbolChange **change );

  /** Insert a python correlation function (assumed to be a dictionary) */
  void insert( PyObject *py_cf, SymbolChange *symb_change );
  void insert( cf &new_cf, SymbolChange *symb_change );

  /** Clears the history */
  void clear();

  /** Returns the number of currently stored entries*/
  unsigned int history_size(){ return buffer_size; };

  /** Returns the index of which the next element will be placed */
  unsigned int get_current_active_positions(){ return current; };
private:
  static const unsigned int max_history = 10;
  std::array<cf*,max_history> cf_history;
  std::array<SymbolChange*,max_history> changes;
  unsigned int current{0};
  unsigned int buffer_size{0};

  void init_all_keys( cf &entry, const std::vector<std::string> &cluster_names );
};
#endif
