#include <cmath>

template<class key,class value>
std::ostream& operator <<( std::ostream &out, const std::map<key,value> &map )
{
  for ( auto iter=map.begin(); iter != map.end(); ++iter )
  {
    out << iter->first << ":" << iter->second << "\n";
  }
  return out;
}
