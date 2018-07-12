#include "eshelby_sphere.hpp"
#include "additional_tools.hpp"
#include <iostream>
using namespace std;

double EshelbySphere::operator()(int i, int j, int k, int l)
{
  return get(i, j, k, l);
}

double EshelbySphere::get(int i, int j, int k, int l) const
{
  double term1 = (5.0*poisson-1.0)*kronecker(i,j)*kronecker(k,l);
  double term2 = (4.0-5.0*poisson)*(kronecker(i,k)*kronecker(j,l) + kronecker(i,l)*kronecker(j,k));
  return (term1+term2)/(15.0*(1-poisson));
}

void EshelbySphere::construct_full_tensor()
{
  for (unsigned int i=0;i<3;i++)
  for (unsigned int j=0;j<3;j++)
  for (unsigned int k=0;k<3;k++)
  for (unsigned int l=0;l<3;l++)
  {
    int indx = get_array_indx(i, j, k, l);
    tensor[indx] = get(i, j, k, l);
  }
}
