#ifndef CF_HISTORY_TRACKER_H
#define CF_HISTORY_TRACKER_H
#include <string>
#include <array>
#include <Python.h>
#include <map>

typedef std::map<std::string,double> cf;

class CFHistoryTracker
{
public:
  CFHistoryTracker();
  ~CFHistoryTracker();

  /** Return a pointer to the next pointer to be written to */
  void get_next( cf *next_cf, PyObject* change );

  /** Returns a reference to the active correlation function */
  cf& get_current(){ return *cf_history[current]; };

  /** Gets the system change and previus */
  void pop( PyObject *change );

  /** Insert a python correlation function (assumed to be a dictionary) */
  void insert( PyObject *py_cf, PyObject* symb_changes );
  void insert( cf &new_cf, PyObject *symb_changes );

  /** Clears the history */
  void clear(){ current=0; buffer_size=0; };

  /** Returns the number of currently stored entries*/
  unsigned int history_size(){ return buffer_size; };
private:
  static const unsigned int max_history = 10;
  std::array<cf*,max_history> cf_history;
  std::array<PyObject*,max_history> changes;
  unsigned int current{0};
  unsigned int buffer_size{0};
};
#endif
