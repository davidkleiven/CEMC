#ifndef CF_HISTORY_TRACKER_H
#define CF_HISTORY_TRACKER_H
#include <string>
#include <array>
#include <Python.h>

typedef std::map<std::string,double> cf;

class CFHistoryTracker
{
public:
  CFHistoryTracker();

  /** Return a pointer to the next pointer to be written to */
  void get_next( cf* next_cf, PyObject* change );

  /** Gets the system change and previus */
  void get_previous( PyObject *change, cf* corrFunc );
private:
  static const unsigned int max_history = 10;
  std::array<cf,max_history> cf_history;
  std::array<PyObject*,max_history> changes;
  unsigned int current{0};
};
#endif
