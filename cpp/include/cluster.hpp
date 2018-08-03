#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <Python.h>

typedef std::vector< std::vector<int> > cluster_t;
typedef std::vector< std::vector<int> > equiv_deco_t;
typedef std::map<std::string, equiv_deco_t > all_equiv_deco_t;

class Cluster
{
public:
  Cluster():size(0),name("noname"){};
  Cluster(PyObject *info_dict);

  /** Returns the built in list */
  const cluster_t& get() const { return members; };
  const cluster_t& get_order() const { return order; };
  const cluster_t& get_equiv() const { return equiv_sites; };
  unsigned int get_size() const {return size;};
  unsigned int num_subclusters() const {return members.size();};
  const equiv_deco_t& get_equiv_deco(const std::string &dec_string) const;
  const equiv_deco_t& get_equiv_deco(const std::vector<int> &deco) const;

/** Finds all the equivalent decoration numbers */
  void construct_equivalent_deco(int n_basis_funcs);

  /** Public attributes */
  std::string name;
  int size;
  unsigned int ref_indx;
  unsigned int symm_group;
  std::string max_cluster_dia;
  std::string descriptor;
private:
  cluster_t members;
  cluster_t order;
  cluster_t equiv_sites;
  all_equiv_deco_t equiv_deco;


  void all_deco(int n_bfs, std::vector< std::vector<int> > &all_deco) const;

  static void deco2string(const std::vector<int> &deco, std::string &name);

  static void nested_list_to_cluster(PyObject *py_list, cluster_t &vec);

  /** Initialize members based on the info dictionary */
  void parse_info_dict(PyObject *info_dict);
};

std::ostream& operator << ( std::ostream& out, const Cluster& clust );
#endif
