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
  #else
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

SymbolChange& py_tuple_to_symbol_change( PyObject *single_change, SymbolChange &symb_change )
{
  symb_change.indx = py2int( PyTuple_GetItem(single_change,0) );
  symb_change.old_symb = py2string( PyTuple_GetItem(single_change,1) );
  symb_change.new_symb = py2string( PyTuple_GetItem(single_change,2) );
  return symb_change;
}

void py_changes2symb_changes( PyObject* all_changes, vector<SymbolChange> &symb_changes )
{
  int size = PyList_Size(all_changes);
  for (unsigned int i=0;i<size;i++ )
  {
    SymbolChange symb_change;
    py_tuple_to_symbol_change( PyList_GetItem(all_changes,i), symb_change );
    symb_changes.push_back(symb_change);
  }
}

void py_change2swap_move(PyObject* all_changes, swap_move &symb_changes)
{
  int size = PyList_Size(all_changes);
  for (unsigned int i=0;i<size;i++ )
  {
    SymbolChange symb_change;
    py_tuple_to_symbol_change( PyList_GetItem(all_changes,i), symb_change );
    symb_changes[i]  = symb_change;
  }
}