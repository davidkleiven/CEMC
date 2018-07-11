#ifndef ESHELBY_TENSOR_H
#define ESHELBY_TENSOR_H
#include <array>
#include <map>
#include <string>

class EshelbyTensor
{
public:
  EshelbyTensor(double a, double b, double c, double poisson);
  virtual ~EshelbyTensor(){};

  /** Evaluate the Eshelby Tensor (Not implemented!)*/
  virtual double operator()(int i, int j, int k, int l);

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
  bool require_rebuild{true};

  void I_tensor(std::array<double,6> &I) const;
  double evlauate_principal(int i, int j, int k, int l) const;
  double tensor[81]; // Rank 4 Eshelby tensor

  /** Return the corresponding index to the array */
  static int get_array_indx(int i, int j, int k, int l);

  /** Shift array circular */
  static void circular_shift(double data[], int size);

  /** The tensor has some symmetries so we only compute
  certain combinations */
  static void sort_indices(int indies[4]);

  /** Convert dictionary key to arrya of indices */
  static void key_to_array(const std::string &key, int array[4]);

  /** Constructs the full Eshelby tensor */
  void construct_full_tensor();

  /** Construct elements that can use the same permuation of the semi axes */
  void construct_ref_tensor( std::map<std::string, double> &elm, double semi_axes[3]);

  /** Elliptic integrals */
  virtual double F(double theta, double kappa) const;
  virtual double E(double theta, double kappa) const;
  virtual double elliptic_integral_python(double theta, double kappa, const char* func_name) const;
};
#endif
