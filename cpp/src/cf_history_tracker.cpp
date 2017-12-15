#include "cf_history_tracker.hpp"

void CFHistoryTracker::get_next( cf *next_cf, PyObject *next_change )
{
  next_cf = &cf_history[current];
  next_change = changes[current];
  current += 1;
  current%max_history;
  return next;
}

void get_last( cf *prev_cf, PyObject *prev_change )
{
  if ( current == 0 )
  {
    current = max_history-1;
  }
  else
  {
    current -= 1;
  }
  prev_cf = &cf_history[current];
  prev_change = changes[current];
}
