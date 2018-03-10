#include "cluster.hpp"
#include "additional_tools.hpp"

using namespace std;
Cluster::Cluster( const string &name, const vector< vector<int> > &mems ):name(name), members(mems)
{
  size = mems[0].size();
}

ostream& operator <<( ostream& out, const Cluster &cluster )
{
  out << "Name: " << cluster.name << "\n";
  out << "Members:\n";
  out << cluster.get();
}
