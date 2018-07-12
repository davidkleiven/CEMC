#ifndef ESHELBY_TENSOR_H
#define ESHELBY_TENSOR_H
#include <array>
#include <map>
#include <string>

typedef std::array< std::array<double, 3>, 3> mat3x3;
typedef std::array<double, 3> vec3;

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

  void I_matrix(mat3x3 &I, vec3 &vec) const;

  void I_matrix_general(mat3x3 &I, const vec3 &vec) const;

  /** Compute the principal vecotor in the general case */
  void I_principal_general(vec3 &vec) const;

  void I_matrix_oblate_sphere(mat3x3 &I, const vec3 &vec) const;

  /** Compute the principal vecotor in the general case */
  void I_principal_oblate_sphere(vec3 &vec) const;

  void I_matrix_prolate_sphere(mat3x3 &I, const vec3 &vec) const;

  /** Compute the principal vecotor in the general case */
  void I_principal_prolate_sphere(vec3 &vec) const;


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

  /** Symmetrize the matrix. Put upper part into lower */
  static void symmetrize(mat3x3 &mat);

  /** Constructs the full Eshelby tensor */
  void construct_full_tensor();

  /** Construct elements that can use the same permuation of the semi axes */
  void construct_ref_tensor( std::map<std::string, double> &elm, const mat3x3 &I, const vec3 &vec, \
    unsigned int shift);

  /** Elliptic integrals */
  virtual double F(double theta, double kappa) const;
  virtual double E(double theta, double kappa) const;
  virtual double elliptic_integral_python(double theta, double kappa, const char* func_name) const;
};
#endif
