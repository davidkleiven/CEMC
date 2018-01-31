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

template<class T>
std::ostream& operator << (std::ostream &out, const std::vector<T> &vec )
{
  for ( unsigned int i=0;i<vec.size(); i++ )
  {
    out << vec[i] << " ";
  }
  return out;
}
