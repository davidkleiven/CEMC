#include <Python.h>
#include "numpy/ndarrayobject.h"

#if PY_MAJOR_VERSION >= 3
int init_numpy()
{
  import_array();
};
#else
void init_numpy()
{
  import_array();
};
#endif
