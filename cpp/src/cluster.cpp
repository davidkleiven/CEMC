#include "cluster.hpp"
#include "additional_tools.hpp"

using namespace std;
Cluster::Cluster( const string &name, const cluster &mems, \
    const cluster &order, const cluster &equiv ):name(name), members(mems), \
      order(order), equiv_sites(equiv)
{
  size = mems[0].size()+1;
}

ostream& operator <<( ostream& out, const Cluster &cluster )
{
  out << "Name: " << cluster.name << "\n";
  out << "Members:\n";
  out << cluster.get();
  out << "\nOrder:\n",
  out << cluster.get_order();
  out << "\nEquivalent sites:\n";
  out << cluster.get_equiv();
}
