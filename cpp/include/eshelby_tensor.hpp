#ifndef ESHELBY_TENSOR_H
#define ESHELBY_TENSOR_H
#include <array>

class EshelbyTensor
{
public:
  EshelbyTensor(double a, double b, double c, double poisson);

  /** Evaluate the Eshelby Tensor */
  virtual double operator()(int i, int j, int k, int l) const;

protected:
  double a;
  double b;
  double c;
  double poisson;
  double elliptic_f;
  double elliptic_e;
  std::array<double,6> I;

  void I_tensor(std::array<double,6> &I) const;
  double evlauate_principal(int i, int j, int k, int l) const;

  /** Elliptic integrals */
  static double F(double theta, double kappa);
  static double E(double theta, double kappa);
  static double elliptic_integral_python(double theta, double kappa, const char* func_name);
};
#endif
