//use_numpy.h

#define PY_ARRAY_UNIQUE_SYMBOL PHASEFIELD_CXX_ARRAY_API 
           
#ifndef INIT_NUMPY_ARRAY_CPP 
    #define NO_IMPORT_ARRAY //for usual translation units
#endif

#include "numpy/arrayobject.h"