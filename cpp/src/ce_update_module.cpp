#include <Python.h>
#include "ce_updater.hpp"
#include <Python.h>
#include <numpy/ndarrayobject.h>
using namespace std;

/**
* Initialize the updater object
*/
static PyObject *get_updater( PyObject *self, PyObject *args )
{
  PyObject *BC;
  PyObject *corrFunc;

  if ( !PyArg_ParseTuple( args, "OO", &BC, &corrFunc) )
  {
    PyErr_SetString( PyExc_TypeError, "Could not parse the supplied arguments!" );
    return NULL;
  }
  PyObject *corr_func_tracker = CEUpdater
}

static PyMethodDef ce_update_module_methods[] = {
  {"get_updater", get_updater, METH_VARARGS, "Get the C++ updater object"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef ce_updater = {
    PyModuleDef_HEAD_INIT,
    "ce_updater",
    NULL, // TODO: Write documentation string here
    -1,
    ce_update_module_methods
  };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_pystructcomp_cpp(void)
  {
    PyObject* module = PyModule_Create( &ce_updater );
    import_array();
    return module;
  };
#else
  PyMODINIT_FUNC initce_updater(void)
  {
    Py_InitModule3( "ce_updater", ce_update_module_methods, "This the Python 2 version" );
    import_array();
  };
#endif
