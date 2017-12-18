#include "cf_history_tracker.hpp"
#include <iostream>

using namespace std;

CFHistoryTracker::CFHistoryTracker()
{
  for ( unsigned int i=0;i<max_history;i++ )
  {
    cf_history[i] = new cf;
    changes[i] = nullptr;
  }
}

CFHistoryTracker::~CFHistoryTracker()
{
  for ( unsigned int i=0;i<max_history;i++ )
  {
    delete cf_history[i];
  }
}

void CFHistoryTracker::get_next( cf *next_cf, PyObject *next_change )
{
  next_cf = cf_history[current];
  next_change = changes[current];
  current += 1;
  current = current%max_history;
}

void CFHistoryTracker::pop( PyObject *prev_change )
{
  if ( current == 0 )
  {
    current = max_history-1;
  }
  else
  {
    current -= 1;
  }

  prev_change = changes[current];
  if ( buffer_size > 0 )
  {
    buffer_size -= 1;
  }
}

void CFHistoryTracker::insert( PyObject *pycf, PyObject *symb_changes )
{
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while(  PyDict_Next(pycf, &pos, &key,&value) )
  {
    (*cf_history[current])[PyString_AsString(key)] = PyFloat_AS_DOUBLE(value);
  }
  changes[current] = symb_changes;
  current++;
  current = current%max_history;

  if ( buffer_size < max_history )
  {
    buffer_size += 1;
  }
}

void CFHistoryTracker::insert( cf &new_cf, PyObject* symb_changes )
{
  for ( auto iter=new_cf.begin(); iter != new_cf.end(); ++iter )
  {
    (*cf_history[current])[iter->first] = iter->second;
  }
  changes[current] = symb_changes;
  current++;
  current = current%max_history;

  if ( buffer_size < max_history )
  {
    buffer_size += 1;
  }
}
