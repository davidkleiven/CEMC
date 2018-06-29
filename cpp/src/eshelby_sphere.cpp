#include "eshelby_sphere.hpp"
#include "additional_tools.hpp"

double EshelbySphere::operator()(int i, int j, int k, int l) const
{
  double term1 = (5.0*poisson-1.0)*kronecker(i,j)*kronecker(k,l)/(15.0*(1-poisson));
  double term2 = (4.0-5.0*poisson)*(kronecker(i,k)*kronecker(j,l) + kronecker(i,l)*kronecker(j,k))/(15.0*(1-poisson));
  return term1+term2;
}
