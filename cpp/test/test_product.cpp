#include "additional_tools.hpp"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
  vector<double> test_vec;
  for ( unsigned int i=0;i<4;i++ )
  {
    test_vec.push_back(i);
  }

  vector< vector<double> > out;
  additional_tools::product( test_vec, out, 4 );

  for ( unsigned int i=0;i<out.size();i++ )
  {
    cout << "[";
    for ( unsigned int j=0;j<out[i].size();j++ )
    {
      cout << out[i][j] << " ";
    }
    cout << "]\n";
  }
  return 0;
}
