#include "cluster.hpp"
#include "additional_tools.hpp"
#include <Python.h>
#include <algorithm>
#include <sstream>
#include <iostream>

using namespace std;

Cluster::Cluster(PyObject *info_dict)
{
  parse_info_dict(info_dict);
}

ostream& operator <<( ostream& out, const Cluster &cluster )
{
  out << "Name: " << cluster.name << "\n";
  out << "Descriptor: " << cluster.descriptor << "\n";
  out << "Max cluster dia: " << cluster.max_cluster_dia << "\n";
  out << "Size: " << cluster.size << "\n";
  out << "ref_indx: " << cluster.ref_indx << "\n";
  out << "Members:\n";
  out << cluster.get();
  out << "\nOrder:\n",
  out << cluster.get_order();
  out << "\nEquivalent sites:\n";
  out << cluster.get_equiv();
  return out;
}

void Cluster::deco2string(const vector<int> &deco, string &name)
{
  stringstream ss;
  for (unsigned int i=0;i<deco.size();i++ )
  {
    ss << deco[i];
  }
  name = ss.str();
}

void Cluster::construct_equivalent_deco(int n_basis_funcs)
{
  vector< vector<int> > bf_indx;
  all_deco(n_basis_funcs, bf_indx);
  if (equiv_sites.size() == 0)
  {
    // There are no equivalent sites, or there are only one basis function
    // Fill the lookup with empty vectors
    string deco_str;
    for (vector<int>& deco : bf_indx)
    {
      deco2string(deco, deco_str);
      vector< vector<int> > one_vector = {deco};
      equiv_deco[deco_str] = one_vector;
    }
    return;
  }


  // Convert the equivalent sites to a list of lists
  PyObject* py_eq_sites = PyList_New(equiv_sites.size());
  for (unsigned int i=0;i<equiv_sites.size();i++)
  {
    if (equiv_sites[i].size() == 0)
    {
      throw runtime_error("One of the entries in equiv_sites are zero!");
    }

    PyObject* py_group = PyList_New(equiv_sites[i].size());
    for (unsigned int j=0;j<equiv_sites[i].size();j++)
    {
      PyObject* py_site = int2py(equiv_sites[i][j]);
      PyList_SetItem(py_group, j, py_site);
    }
    PyList_SetItem(py_eq_sites, i, py_group);
  }

  // Use the python methods due to the convenient itertools module
  string mod_name("ase.ce.tools");
  PyObject *mod_string = string2py(mod_name);
  PyObject *ce_tools_mod = PyImport_Import(mod_string);
  PyObject *equiv_deco_func = PyObject_GetAttrString(ce_tools_mod, "equivalent_deco");

  if (equiv_deco_func == nullptr)
  {
    throw runtime_error("Could not import equivalanet_deco function!");
  }

  for (vector<int>& deco : bf_indx)
  {
    string deco_str;
    deco2string(deco, deco_str);

    // Convert the vector of int into a python list
    PyObject *dec_list = PyList_New(deco.size());
    for (unsigned int i=0;i<deco.size();i++)
    {
      PyObject *py_int = int2py(deco[i]);
      PyList_SetItem(dec_list, i, py_int);
    }

    PyObject *args = PyTuple_Pack(2, dec_list, py_eq_sites);
    PyObject* py_equiv = PyObject_CallObject(equiv_deco_func, args);

    // Create a nested vector based on the result from Python
    vector< vector<int> > eq_dec;
    int size = PyList_Size(py_equiv);
    for (unsigned int i=0;i<size;i++)
    {
      vector<int> one_dec;
      PyObject *py_one_dec = PyList_GetItem(py_equiv, i);
      for (unsigned int j=0;j<deco.size();j++)
      {
        one_dec.push_back( py2int(PyList_GetItem(py_one_dec, j)) );
      }
      eq_dec.push_back(one_dec);
    }

    equiv_deco[deco_str] = eq_dec;
    Py_DECREF(py_equiv);
    Py_DECREF(args);
    Py_DECREF(dec_list);
  }

  Py_DECREF(mod_string);
  Py_DECREF(ce_tools_mod);
  Py_DECREF(equiv_deco_func);
  Py_DECREF(py_eq_sites);
}


void Cluster::all_deco(int num_bfs, vector< vector<int> > &deco) const
{
  if (get_size() <= 1)
  {
    return;
  }
  else if (get_size() == 2)
  {
    for (unsigned int i=0;i<num_bfs;i++)
    for (unsigned int j=0;j<num_bfs;j++)
    {
      vector<int> vec = {i, j};
      deco.push_back(vec);
    }
  }
  else if (get_size() == 3)
  {
    for (unsigned int i=0;i<num_bfs;i++)
    for (unsigned int j=0;j<num_bfs;j++)
    for (unsigned int k=0;k<num_bfs;k++)
    {
      vector<int> vec = {i, j, k};
      deco.push_back(vec);
    }
  }
  else if(get_size() == 4)
  {
    for (unsigned int i=0;i<num_bfs;i++)
    for (unsigned int j=0;j<num_bfs;j++)
    for (unsigned int k=0;k<num_bfs;k++)
    for (unsigned int l=0;l<num_bfs;l++)
    {
      vector<int> vec = {i, j, k, l};
      deco.push_back(vec);
    }
  }
  else
  {
    throw invalid_argument("Only cluster sizes 2, 3 and 4 are supported!");
  }
}

const equiv_deco_t& Cluster::get_equiv_deco(const string& dec_str) const
{
  return equiv_deco.at(dec_str);
}

const equiv_deco_t& Cluster::get_equiv_deco(const std::vector<int> &deco) const
{
  string dec_str;
  deco2string(deco, dec_str);
  return get_equiv_deco(dec_str);
}

void Cluster::parse_info_dict(PyObject *info)
{
  // Read reference index
  PyObject* py_ref_indx = PyDict_GetItemString(info, "ref_indx");
  ref_indx = py2int(py_ref_indx);

  // Read size
  PyObject* py_size = PyDict_GetItemString(info, "size");
  size = py2int(py_size);

  // Read max_cluster_dia
  PyObject* py_mx_dia = PyDict_GetItemString(info, "max_cluster_dia");
  if (size <= 1)
  {
    max_cluster_dia = "none";
  }
  else
  {
    max_cluster_dia = py2string(py_mx_dia);
  }
  // Read symmetry group
  PyObject* py_symm = PyDict_GetItemString(info, "symm_group");
  symm_group = py2int(py_symm);


  // Read the name
  PyObject* py_name = PyDict_GetItemString(info, "name");
  name = py2string(py_name);

  // Read descriptor
  PyObject* py_desc = PyDict_GetItemString(info, "descriptor");
  descriptor = py2string(py_desc);

  // Read indices
  PyObject* py_indx = PyDict_GetItemString(info, "indices");
  nested_list_to_cluster(py_indx, members);

  // Read the order
  if (size >= 2)
  {
    PyObject* py_order = PyDict_GetItemString(info, "order");
    nested_list_to_cluster(py_order, order);
  }

  // Read equivalent sites
  PyObject *py_equiv_sites = PyDict_GetItemString(info, "equiv_sites");
  nested_list_to_cluster(py_equiv_sites, equiv_sites);
}

void Cluster::nested_list_to_cluster(PyObject *py_list, cluster_t &vec)
{
  int size = PyList_Size(py_list);
  for (int i=0;i<size;i++)
  {
    vector<int> one_cluster;
    PyObject* seq = PySequence_Fast(PyList_GetItem(py_list, i), NULL); // New reference
    int n_memb = PySequence_Fast_GET_SIZE(seq);
    for (int j=0;j<n_memb;j++)
    {
      one_cluster.push_back(py2int(PySequence_Fast_GET_ITEM(seq, j)));
    }
    Py_DECREF(seq);
    vec.push_back(one_cluster);
  }
}
