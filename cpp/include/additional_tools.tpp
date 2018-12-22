#include <cmath>

template<class key,class value>
std::ostream& operator <<( std::ostream &out, const std::map<key,value> &map )
{
  for ( auto iter=map.begin(); iter != map.end(); ++iter )
  {
    out << iter->first << ":" << iter->second << "\n";
  }
  return out;
};

template<class T>
std::ostream& operator << (std::ostream &out, const std::vector<T> &vec )
{
  out << "[";
  for ( unsigned int i=0;i<vec.size(); i++ )
  {
    out << vec[i] << " ";
  }
  out << "]";
  return out;
};

template<class T>
std::vector<T>& cyclic_permute( std::vector<T> &vec )
{
  for ( unsigned int i=0;i<vec.size()-1;i++ )
  {
    T prev = vec[0];
    vec[0] = vec[i+1];
    vec[i+1] = prev;
  }
  return vec;
};

template<class T>
void keys( std::map<std::string,T> &dict, std::vector<std::string> &names )
{
  names.clear();
  for ( auto iter=dict.begin(); iter != dict.end(); ++iter )
  {
    names.push_back(iter->first);
  }
};

template<class T>
void set2vector( const std::set<T> &s, std::vector<T> &vec )
{
  vec.clear();
  for ( auto iter = s.begin(); iter != s.end(); ++iter )
  {
    vec.push_back(*iter);
  }
};

template<class T, unsigned int N>
std::ostream& operator <<(std::ostream& out, std::array<T, N> &array)
{
  for (unsigned int i=0;i<N;i++ )
  {
    out << array[i];
  }
  out << "\n";
  return out;
};

template<class T>
std::ostream& operator <<(std::ostream& out, std::set<T> &set){
  for (auto iter=set.begin(); iter != set.end(); ++iter){
    out << *iter << " ";
  }
  out << "\n";
  return out;
}

template<class T>
bool is_in_vector(const T& value, const std::vector<T> &vector){
  for (const T& elem : vector){
    if (value == elem) return true;
  }
  return false;
}

template<class T>
void insert_in_set(const std::vector<T> &vec, std::set<T> &unique){
  for (const T& elem : vec){
    unique.insert(elem);
  }
}