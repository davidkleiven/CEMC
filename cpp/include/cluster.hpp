#ifndef CLUSTER_H
#define CLUSTER_H
#include <vector>
#include <string>
#include <memory>
#include <iostream>

class Cluster
{
public:
  Cluster():size(0),name("noname"){};
  Cluster( const std::string &name, const std::vector< std::vector<int> > &members );

  /** Returns the built in list */
  const std::vector< std::vector<int> >& get() const { return members; };

  /** Public attributes */
  std::string name;
  int size;
private:
  std::vector< std::vector<int> > members;
};

std::ostream& operator << ( std::ostream& out, const Cluster& clust );
#endif
