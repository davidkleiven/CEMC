#include "cf_history_tracker.hpp"
#include "additional_tools.hpp"
#include <iostream>

using namespace std;

CFHistoryTracker::CFHistoryTracker( const vector<string> &cluster_names )
{
  for ( unsigned int i=0;i<max_history;i++ )
  {
    cf_history[i] = new cf;
    changes[i] = new SymbolChange;
    cf_history[i]->set_order(cluster_names);
    //init_all_keys(*cf_history[i],cluster_names);
  }
}

CFHistoryTracker::CFHistoryTracker( const CFHistoryTracker &other )
{
  swap(*this,other);
}

CFHistoryTracker& CFHistoryTracker::operator=( const CFHistoryTracker &other )
{
  swap(*this,other);
  return *this;
}

CFHistoryTracker::~CFHistoryTracker()
{
  for ( unsigned int i=0;i<max_history;i++ )
  {
    delete cf_history[i];
    delete changes[i];
  }
}

void CFHistoryTracker::init_all_keys( cf &entry, const vector<string> &cluster_names )
{
  for ( unsigned int i=0;i<cluster_names.size();i++ )
  {
    entry[cluster_names[i]] = 0.0;
  }
}

void CFHistoryTracker::get_next( cf **next_cf, SymbolChange **next_change )
{
  *next_cf = cf_history[current];
  *next_change = changes[current];
  current += 1;
  current = current%max_history;
  if ( buffer_size < max_history )
  {
    buffer_size += 1;
  }
}

void CFHistoryTracker::pop( SymbolChange **prev_change )
{
  if ( buffer_size == 0 )
  {
    *prev_change = nullptr;
    return;
  }

  if ( current == 0 )
  {
    current = max_history-1;
  }
  else
  {
    current -= 1;
  }

  *prev_change = changes[current];
  if ( buffer_size > 0 )
  {
    buffer_size -= 1;
  }
}

void CFHistoryTracker::insert( PyObject *pycf, SymbolChange *symb_changes )
{
  Py_ssize_t pos = 0;
  PyObject *key;
  PyObject *value;
  while(  PyDict_Next(pycf, &pos, &key,&value) )
  {
    string new_key = PyString_AsString(key);
    if ( cf_history[current]->count(new_key) == 0 )
    {
      continue;
    }
    (*cf_history[current])[new_key] = PyFloat_AS_DOUBLE(value);
  }

  if ( symb_changes != nullptr )
  {
    *changes[current] = *symb_changes;
  }
  current++;
  current = current%max_history;

  if ( buffer_size < max_history )
  {
    buffer_size += 1;
  }
}

void CFHistoryTracker::insert( cf &new_cf, SymbolChange* symb_changes )
{
  //for ( auto iter=new_cf.begin(); iter != new_cf.end(); ++iter )
  for ( unsigned int i=0;i<new_cf.size();i++ )
  {
    (*cf_history[current])[i] = new_cf[i];
  }

  if ( symb_changes != nullptr )
  {
    *changes[current] = *symb_changes;
  }

  current++;
  current = current%max_history;

  if ( buffer_size < max_history )
  {
    buffer_size += 1;
  }
}

cf& CFHistoryTracker::get_current()
{
  unsigned int indx = current;
  if ( current == 0 )
  {
    indx = max_history-1;
  }
  else
  {
    indx = current-1;
  }
  return *cf_history[indx];
}

void CFHistoryTracker::clear()
{
  buffer_size = 1;
}

void swap( CFHistoryTracker &first, const CFHistoryTracker &second )
{
  for ( unsigned int i=0;i<second.max_history;i++ )
  {
    first.changes[i] = new SymbolChange;
    first.cf_history[i] = new cf;
    *first.changes[i] = *second.changes[i];
    *first.cf_history[i] = *second.cf_history[i];
  }
  first.current = second.current;
  first.buffer_size = second.buffer_size;
}
