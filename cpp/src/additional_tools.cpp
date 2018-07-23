#include "additional_tools.hpp"
#include "cf_history_tracker.hpp"

using namespace std;

ostream& operator<<(ostream& out, const SymbolChange &symb )
{
  out << "(Index: " << symb.indx << " old symbol: " << symb.old_symb << " new symbol: " << symb.new_symb << " track index: " << symb.track_indx << ")";
  return out;
}

std::ostream& operator <<(ostream &out, const array<SymbolChange,2> &move )
{
  out << move[0] << "->" << move[1];
  return out;
}

int kronecker(int i, int j)
{
  if (i==j) return 1;
  return 0;
};

PyObject* string2py(const string &str)
{
  #if PY_MAJOR_VERSION >= 3
    // Python 3
    return PyUnicode_FromString(str.c_str());
  #else:
    // Python 2
    return PyUnicode_FromString(str.c_str());
  #endif
}

string py2string(PyObject *str)
{
  #if PY_MAJOR_VERSION >= 3
    // Python 3
    return PyUnicode_AsUTF8(str);
  #else
    // Python 2
    return PyString_AsString(str);
  #endif
}

PyObject *int2py(int integer)
{
  #if PY_MAJOR_VERSION >= 3
    return PyLong_FromLong(integer);
  #else
    return PyInt_FromLong(integer);
  #endif
}

int py2int(PyObject *integer)
{
  #if PY_MAJOR_VERSION >= 3
    return PyLong_AsLong(integer);
  #else
    return PyInt_AsLong(integer);
  #endif
}
