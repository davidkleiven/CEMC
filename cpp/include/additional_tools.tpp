#include <cmath>
#include <vector>
void additional_tools::product( std::vector<T> &in, std::vector< std::vector<T> > &out, unsigned int repeat )
{
  unsigned int n = pow( in.size(), repeat );
  std::vector<T> vec(repeat);
  std::vector<unsigned int> indices(repeat);
  for ( unsigned int i=0;i<n;i++ )
  {
    unsigned int indx = i%repeat;
    vec[counter++] = in[indx];
    for ( unsigned int j=1;j<repeat;j++ )
    {
      unsigned int index = i/pow(repeat,j);
      vec[counter++] = in[indx];
    }
    out.push_back(vec);
    counter = 0;
  }
};
