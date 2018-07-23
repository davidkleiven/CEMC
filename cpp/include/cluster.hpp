#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>
#include <string>
#include <memory>
#include <iostream>

typedef std::vector< std::vector<int> > cluster;
class Cluster
{
public:
  Cluster():size(0),name("noname"){};
  Cluster( const std::string &name, const cluster &members, const cluster &order, const cluster &equiv );

  /** Returns the built in list */
  const cluster& get() const { return members; };
  const cluster& get_order() const { return order; };
  const cluster& get_equiv() const { return equiv_sites; };

  /** Public attributes */
  std::string name;
  int size;
private:
  cluster members;
  cluster order;
  cluster equiv_sites;
};

std::ostream& operator << ( std::ostream& out, const Cluster& clust );
#endif
