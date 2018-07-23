#ifndef INIT_NUMPY_H
#define INIT_NUMPY_H

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL CE_UPDATER_ARRAY_API
#include "numpy/ndarrayobject.h"

#if PY_MAJOR_VERSION >= 3
inline int* init_numpy()
{
  import_array();
   // Avoid compilation warning
   // import_array is a macro that inserts a return statement
   // so it has no practical effect
  return NULL;
};
#else
inline void init_numpy()
{
  import_array();
};
#endif
#endif
