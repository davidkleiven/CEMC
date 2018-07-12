#ifndef ESHELBY_SPHERE_H
#define ESHELBY_SPHERE_H
#include "eshelby_tensor.hpp"

class EshelbySphere: public EshelbyTensor
{
public:
  EshelbySphere(double a, double poisson):EshelbyTensor(a, a, a, poisson){};
  virtual ~EshelbySphere(){};

  /** Acces operator */
  virtual double operator()(int i, int j, int k, int l) override;

  /** Alternative access */
  double get(int i, int j, int k, int l) const;

  /** Nothing to initialize for a sphere */
  virtual void init() override {}; // No initialization is needed here
private:
  virtual void construct_full_tensor() override;
};
#endif
