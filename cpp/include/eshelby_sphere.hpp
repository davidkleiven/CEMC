#ifndef ESHELBY_SPHERE_H
#define ESHELBY_SPHERE_H
#include "eshelby_tensor.hpp"

class EshelbySphere: public EshelbyTensor
{
public:
  EshelbySphere(double a, double poisson):EshelbyTensor(a, a, a, poisson){};
  virtual ~EshelbySphere(){};

  /** Acces operator */
  virtual double operator()(int i, int j, int k, int l) const override;

  /** Nothing to initialize for a sphere */
  virtual void init() override {}; // No initialization is needed here
private:
  double a;
  double poisson;
};
#endif
