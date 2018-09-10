#define INIT_NUMPY_ARRAY_CPP

#include "use_numpy.hpp"
#if PY_MAJOR_VERSION >= 3
int init_numpy()
{
  import_array();
   // Avoid compilation warning
   // import_array is a macro that inserts a return statement
   // so it has no practical effect
  return 0;
};
#else
void void_import_array(){
  import_array();
}

int init_numpy()
{
  void_import_array();
  return 0;
};
#endif

const static int numpy_initialized = init_numpy();
