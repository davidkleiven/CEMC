#include "eshelby_cylinder.hpp"
#include <cmath>

double EshelbyCylinder::operator()(int i, int j, int k, int l)
{
  int indices[4] = {i,j,k,l};
  sort_indices(indices);
  i = indices[0];
  j = indices[1];
  k = indices[2];
  l = indices[3];

  double pref = 0.5/(1-poisson);
  double two_poiss = 1.0-2*poisson;

  if ((i==1) && (j==1) && (k==1) && (l==1))
  {
    double term1 = (b*b+2*a*b)/pow(a+b,2);
    double term2 = b*two_poiss/(a+b);
    return pref*(term1+term2);
  }
  else if ((i==2) && (j==2) && (k==2) && (l==2))
  {
    double term1 = (a*a+2*a*b)/pow(a+b,2);
    double term2 = two_poiss*a/(a+b);
    return pref*(term1+term2);
  }
  else if ((i==1) && (j==1) && (k==2) && (l==2))
  {
    double term1 = b*b/pow(a+b,2);
    double term2 = two_poiss*b/(a+b);
    return pref*(term1-term2);
  }
  else if ((i==2) && (j==2) && (k==3) && (l==3))
  {
    return pref*2*poisson*a/(a+b);
  }
  else if ((i==2) && (j==2) && (k==1) && (l==1))
  {
    double term1 = pow(a/(a+b),2);
    double term2 = two_poiss*a/(a+b);
    return pref*(term1-term2);
  }
  else if ((i==1) && (j==2) && (k==1) && (l==2))
  {
    double term1 = 0.5*(a*a+b*b)/pow(a+b,2);
    double term2 = 0.5*two_poiss;
    return pref*(term1+term2);
  }
  else if ((i==1) && (j==1) && (j==3) && (k==3))
  {
    return pref*2*poisson*b/(a+b);
  }
  else if ((i==2) && (j==3) && (k==2) && (l==3))
  {
    return 0.5*a/(a+b);
  }
  else if ((i==1) && (j==3) && (k==1) && (l==3))
  {
    return 0.5*b/(a+b);
  }
  return 0.0;
}
