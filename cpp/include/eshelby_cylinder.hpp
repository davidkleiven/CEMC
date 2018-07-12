#ifndef ESHELBY_CYLINDER_H
#define ESHELBY_CYLINDER_H
#include "eshelby_tensor.hpp"

class EshelbyCylinder: public EshelbyTensor
{
public:
  EshelbyCylinder(double a, double b, double poisson): EshelbyTensor(a, b, a, poisson){};

  virtual double operator()(int i, int j, int k, int l) override;

  /** No initialization required */
  virtual void init() override{};
};
#endif
