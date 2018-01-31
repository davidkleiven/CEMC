#include "additional_tools.hpp"
#include "cf_history_tracker.hpp"

using namespace std;

ostream& operator<<(ostream& out, const SymbolChange &symb )
{
  out << "Index: " << symb.indx << " old symbol: " << symb.old_symb << " new symbol: " << symb.new_symb << " track index: " << symb.track_indx;
  return out;
}
