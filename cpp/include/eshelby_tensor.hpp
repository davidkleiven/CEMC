#ifndef ESHELBY_TENSOR_H
#define ESHELBY_TENSOR_H
#include <array>

class EshelbyTensor
{
public:
  EshelbyTensor(double a, double b, double c, double poisson);
  virtual ~EshelbyTensor(){};

  /** Evaluate the Eshelby Tensor (Not implemented!)*/
  virtual double operator()(int i, int j, int k, int l) const{return 0.0;};

  /** Initialize elliptic integrals etc. Has to be called prior to evaluation */
  virtual void init();
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

  /** The tensor has some symmetries so we only compute
  certain combinations */
  static void sort_indices( int indies[4]);

  /** Elliptic integrals */
  virtual double F(double theta, double kappa);
  virtual double E(double theta, double kappa);
  virtual double elliptic_integral_python(double theta, double kappa, const char* func_name);
};
#endif
