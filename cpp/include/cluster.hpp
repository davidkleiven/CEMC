#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>

typedef std::vector< std::vector<int> > cluster_t;
typedef std::vector< std::vector<int> > equiv_deco_t;
typedef std::map<std::string, equiv_deco_t > all_equiv_deco_t;

class Cluster
{
public:
  Cluster():size(0),name("noname"){};
  Cluster( const std::string &name, const cluster_t &members, const cluster_t &order, const cluster_t &equiv );

  /** Returns the built in list */
  const cluster_t& get() const { return members; };
  const cluster_t& get_order() const { return order; };
  const cluster_t& get_equiv() const { return equiv_sites; };
  unsigned int get_size() const {return members[0].size()+1;};
  unsigned int num_subclusters() const {return members.size();};
  const equiv_deco_t& get_equiv_deco(const std::string &dec_string) const;
  const equiv_deco_t& get_equiv_deco(const std::vector<int> &deco) const;

/** Finds all the equivalent decoration numbers */
  void construct_equivalent_deco(int n_basis_funcs);

  /** Public attributes */
  std::string name;
  int size;
private:
  cluster_t members;
  cluster_t order;
  cluster_t equiv_sites;
  all_equiv_deco_t equiv_deco;

  void all_deco(int n_bfs, std::vector< std::vector<int> > &all_deco) const;

  static void deco2string(const std::vector<int> &deco, std::string &name);
};

std::ostream& operator << ( std::ostream& out, const Cluster& clust );
#endif
