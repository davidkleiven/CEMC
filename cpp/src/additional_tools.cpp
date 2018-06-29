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
