#include "eshelby_tensor.hpp"
#include "additional_tools.hpp"
#include <Python.h>
#include <cmath>
#include <stdexcept>
#include <map>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

const double PI = acos(-1.0);
EshelbyTensor::EshelbyTensor(double a, double b, double c, double poisson):a(a), b(b), c(c), poisson(poisson)
{
  if ((a < b) || (a < c) || (b<c))
  {
    throw invalid_argument("The Eshelby tensor assumes that a > b > c!");
  }
}

void EshelbyTensor::init()
{
  double theta = asin( sqrt((a*a-c*c)/a*a) );
  double kappa = sqrt( (a*a-b*b)/(a*a-c*c) );
  elliptic_f = F(theta, kappa);
  elliptic_e = E(theta, kappa);
}

double EshelbyTensor::elliptic_integral_python(double theta, double kappa, const char* funcname) const
{
  PyObject* mod_string = string2py("scipy.special");
  PyObject* scipy_spec = PyImport_Import(mod_string);
  PyObject* func = PyObject_GetAttrString(scipy_spec, funcname);
  PyObject* m = PyFloat_FromDouble(kappa*kappa);
  PyObject* py_theta = PyFloat_FromDouble(theta);
  PyObject* args = PyTuple_Pack(2, py_theta, m);
  PyObject *res = PyObject_CallObject(func, args);
  double value = PyFloat_AsDouble(res);

  Py_DECREF(mod_string);
  Py_DECREF(scipy_spec);
  Py_DECREF(func);
  Py_DECREF(m);
  Py_DECREF(py_theta);
  Py_DECREF(args);
  Py_DECREF(res);
  return value;
}

double EshelbyTensor::F(double theta, double kappa) const
{
  return elliptic_integral_python(theta, kappa, "ellipkinc");
}

double EshelbyTensor::E(double theta, double kappa) const
{
  return elliptic_integral_python(theta, kappa, "ellipeinc");
}

void EshelbyTensor::I_matrix(mat3x3 &result, vec3 &princ) const
{
  double tol = 1E-6;
  if (abs(a*a - c*c) < tol)
  {
    throw invalid_argument("Sphere is not implemented yet!");
  }
  else if (abs(a*a - b*b) < tol)
  {
    I_principal_oblate_sphere(princ);
    I_matrix_oblate_sphere(result, princ);
  }
  else if (abs(b*b - c*c) < tol)
  {
    I_principal_prolate_sphere(princ);
    I_matrix_prolate_sphere(result, princ);
  }
  else
  {
    I_principal_general(princ);
    I_matrix_general(result, princ);
  }
  symmetrize(result);
}

void EshelbyTensor::I_principal_general(vec3 &vec) const
{
  double theta = asin(sqrt(1.0 - pow(c/a,2)));
  double kappa = sqrt( (a*a - b*b)/(a*a - c*c));
  double f = F(theta, kappa);
  double e = E(theta, kappa);

  vec[0] = 4.0*PI*a*b*c*(f-e)/((a*a-b*b)*sqrt(a*a-c*c));
  vec[2] = 4.0*PI*a*b*c/((b*b-c*c)*sqrt(a*a-c*c));
  vec[2] *= (b*sqrt(a*a-c*c)/(a*c) - e);
  vec[1] = 4.0*PI - vec[0] - vec[2];
}

void EshelbyTensor::I_matrix_general(mat3x3 &result, const vec3 &princ) const
{
  vec3 semi_axes;
  semi_axes[0] = a*a;
  semi_axes[1] = b*b;
  semi_axes[2] = c*c;

  // Loop over cyclic permutations
  for (unsigned int perm=0;perm<3;perm++)
  {
    unsigned int i1 = perm;
    unsigned int i2 = (perm+1)%3;
    unsigned int i3 = (perm+2)%3;

    result[i1][i2] = (princ[i2]-princ[i1])/(3.0*(semi_axes[i1] - semi_axes[i2]));
    result[i1][i3] = princ[i1] - 4.0*PI/3.0 + \
      (semi_axes[i1] - semi_axes[i2])*result[i1][i2];

    result[i1][i3] /= (semi_axes[i3] - semi_axes[i1]);

    result[i1][i1] = 4.0*PI/(3.0*semi_axes[i1]) - result[i1][i2] - result[i1][i3];
  }
}

void EshelbyTensor::I_principal_oblate_sphere(vec3 &vec) const
{
  // a = b > c
  vec[0] = 2.0*PI*a*a*c*(acos(c/a) - (c/a)*sqrt(1-pow(c/a, 2)))/pow(a*a-c*c, 1.5);
  vec[1] = vec[0];
  vec[2] = 4.0*PI - vec[0] - vec[1];
}

void EshelbyTensor::I_matrix_oblate_sphere(mat3x3 &result, const vec3 &princ) const
{
  // a = b > c
  vec3 semi_axes;
  semi_axes[0] = a*a;
  semi_axes[1] = b*b;
  semi_axes[2] = c*c;
  double tol = 1E-6;

  // First row
  result[0][2] = princ[0] - 4.0*PI/3.0;
  result[0][2] /= (semi_axes[2] - semi_axes[0]);
  result[0][0] = PI/semi_axes[0] - 0.75*result[0][2];
  result[0][1] = result[0][0]/3.0;

  // Second row
  result[1][2] = (princ[2] - princ[1])/(3.0*(semi_axes[1] - semi_axes[2]));
  result[1][1] = 4.0*PI/(3.0*semi_axes[1]) - result[1][2] - result[0][1];
  result[1][0] = result[0][1];

  // Third row
  result[2][0] = result[0][2];
  result[2][1] = result[1][2];
  result[2][2] = 4.0*PI/(3.0*semi_axes[2]) - result[0][2] - result[1][2];
}

void EshelbyTensor::I_principal_prolate_sphere(vec3 &vec) const
{
  // a > b=c
  vec[1] = 2.0*PI*a*c*c*( (a/c)*sqrt(pow(a/c, 2) - 1) - acosh(a/c));
  vec[1] /= pow(a*a-c*c, 1.5);
  vec[2] = vec[1];
  vec[0] = 4.0*PI - vec[1] - vec[2];
}

void EshelbyTensor::I_matrix_prolate_sphere(mat3x3 &result, const vec3 &princ) const
{
  // a > b = c
  vec3 semi_axes;
  semi_axes[0] = a*a;
  semi_axes[1] = b*b;
  semi_axes[2] = c*c;

  // First row
  result[0][1] = (princ[1] - princ[0])/(3.0*(semi_axes[0] - semi_axes[1]));
  result[0][2] = princ[0] - 4.0*PI/3.0 + (semi_axes[0] - semi_axes[1])*result[0][1];
  result[0][2] /= (semi_axes[2] - semi_axes[0]);
  result[0][0] = 4.0*PI/(3.0*semi_axes[0]) - result[0][1] - result[0][2];

  // Second row
  result[1][0] = result[0][1];
  result[1][2] = PI/(3.0*semi_axes[1]) - 0.25*result[0][1];
  result[1][1] = 3.0*result[1][2];

  // Third row
  result[2][0] = result[0][2];
  result[2][1] = result[1][2];
  result[2][2] = (4.0*PI)/(3.0*semi_axes[2]) - result[0][2] - result[1][2];
}


double EshelbyTensor::evlauate_principal(int i, int j, int k, int l) const
{
  // Reference:
  // Weinberger, Chris, and Wei Cai. "Lecture Note 2. Eshelby’s Inclusion I." (2004).
  if ((i==1) && (j==1) && (k==1) && (l==1))
  {
    double I11 = I[3];
    double I1 = I[0];
    double term1 = 3.0*a*a*I11/(8*PI*(1-poisson));
    double term2 = (1-2*poisson)*I1/(8*PI*(1-poisson));
    return term1+term2;
  }
  else if ((i==1) && (j==1) && (k==2) && (l==2))
  {
    double I1 = I[0];
    double I12 = I[4];
    double term1 = b*b*I12/(8*PI*(1-poisson));
    double term2 = (1-2*poisson)*I1/(8*PI*(1-poisson));
    return term1 + term2;
  }
  else if ((i==1) && (j==1) && (k==3) && (l==3))
  {
    double I13 = I[5];
    double I1 = I[0];
    double term1 = c*c*I13/(8*PI*(1-poisson));
    double term2 = (1-2*poisson)*I1/(8*PI*(1-poisson));
    return term1 + term2;
  }
  else if((i==1) && (j==2) && (k==1) && (l==2))
  {
    double I12 = I[4];
    double I1 = I[0];
    double I2 = I[1];
    double term1 = (a*a+b*b)*I12/(16*PI*(1-poisson));
    double term2 = (1-2*poisson)*(I1+I2)/(16*PI*(1-poisson));
    return term1 + term2;
  }
  else if (((i==1) && (j==1) && (k==1) && (l==2)) || \
           ((i==1) && (j==2) && (k==2) && (l==3)) || \
           ((i==1) && (j==2) && (k==3) && (l==2)))
  {
    return 0.0;
  }
  else
  {
    throw invalid_argument("The given combination of ijkl is not a principal combination!");
  }
}

void EshelbyTensor::sort_indices( int orig[4] )
{
  if (orig[0] > orig[1])
  {
    int temp_orig0 = orig[0];
    orig[0] = orig[1];
    orig[1] = temp_orig0;
  }

  if (orig[2] > orig[3])
  {
    int temp_orig2 = orig[2];
    orig[2] = orig[3];
    orig[3] = temp_orig2;
  }
}

double EshelbyTensor::operator()(int i, int j, int k, int l)
{
  if (require_rebuild)
  {
    construct_full_tensor();
  }
  // Sort the indices to take the symmetries of the tensor into account
  int indx_in_array = get_array_indx(i, j, k, l);
  return tensor[indx_in_array];
}

void EshelbyTensor::construct_full_tensor()
{
  for (unsigned int i=0;i<81;i++)
  {
    tensor[i] = 0.0;
  }

  mat3x3 I_mat;
  vec3 princ;
  I_matrix(I_mat, princ);

  for (unsigned int shift=0;shift<3;shift++ )
  {
    map<string, double> ref_tensor;
    construct_ref_tensor(ref_tensor, I_mat, princ, shift);

    for (auto iter=ref_tensor.begin(); iter != ref_tensor.end(); ++iter)
    {
      int indices[4];
      key_to_array(iter->first, indices);

      // Update the array to the current cyclic permutation
      for (unsigned int n=0;n<4;n++)
      {
        indices[n] = (indices[n]+shift)%3;
      }

      int indx_in_array = get_array_indx(indices[0], indices[1], indices[2], indices[3]);
      tensor[indx_in_array] = iter->second;

      // Populate minor symmetries also
      indx_in_array = get_array_indx(indices[0], indices[1], indices[3], indices[2]);
      tensor[indx_in_array] = iter->second;

      indx_in_array = get_array_indx(indices[1], indices[0], indices[3], indices[2]);
      tensor[indx_in_array] = iter->second;

      indx_in_array = get_array_indx(indices[1], indices[0], indices[2], indices[3]);
      tensor[indx_in_array] = iter->second;
    }
  }
  require_rebuild = false;
}

void EshelbyTensor::construct_ref_tensor(map<string, double> &elements, const mat3x3 &I, \
  const vec3 &princ, unsigned int shift)
{
  vec3 semi_axes;
  semi_axes[0] = a*a;
  semi_axes[1] = b*b;
  semi_axes[2] = c*c;

  int i1 = shift;
  int i2 = (shift+1)%3;
  int i3 = (shift+2)%3;

  double pref = 1.0/(8.0*PI*(1-poisson));
  double Q = 3.0*pref;
  double R = (1.0-2*poisson)*pref;

  // S_1111
  elements["0000"] = Q*semi_axes[i1]*I[i1][i1] + R*princ[i1];

  // S_1122
  elements["0011"] = Q*semi_axes[i2]*I[i1][i2] - R*princ[i1];

  // S_1133
  elements["0022"] = Q*semi_axes[i3]*I[i1][i3] - R*princ[i1];

  // S_1212
  elements["0101"] = 0.5*Q*(semi_axes[i1] + semi_axes[i2])*I[i1][i2]\
    + 0.5*R*(princ[i1]+princ[i2]);

  elements["0001"] = 0.0;
  elements["0112"] = 0.0;
  elements["0121"] = 0.0;
}

int EshelbyTensor::get_array_indx(int i, int j, int k, int l)
{
  return i*27 + j*9 + k*3 + l;
}

void EshelbyTensor::key_to_array(const string &key, int array[4])
{
  for (unsigned int i=0;i<4;i++)
  {
    array[i] = key[i] - '0';
  }
}

void EshelbyTensor::array_to_key(string &key, int array[4])
{
  stringstream ss;
  for (unsigned int i=0;i<4;i++)
  {
    ss << array[i];
  }
  key = ss.str();
}
void EshelbyTensor::circular_shift(double data[], int size)
{
  double first = data[0];
  for (unsigned int i=0;i<size-1;i++)
  {
    data[i] = data[i+1];
  }
  data[size-1] = first;
}

void EshelbyTensor::symmetrize(mat3x3 &matrix)
{
  for (unsigned int row=1;row<3;row++)
  {
    for (unsigned int col=0;col<row-1;col++)
    {
      matrix[row][col] = matrix[col][row];
    }
  }
}


PyObject* EshelbyTensor::aslist()
{
  mat6x6 voigt;
  voigt_representation(voigt);
  PyObject *npy_array = PyList_New(6);

  for (unsigned int i=0;i<6;i++)
  {
    PyObject *sublist = PyList_New(6);
    for (unsigned int j=0;j<6;j++)
    {
      PyObject *py_val = PyFloat_FromDouble(voigt[i][j]);

      // NOTE: SetItem steals a reference. So no DECREF needed.
      PyList_SetItem(sublist, j, py_val);
    }
    PyList_SetItem(npy_array, i, sublist);
  }
  return npy_array;
}

void EshelbyTensor::voigt_representation(mat6x6 &voigt)
{
  if (require_rebuild) construct_full_tensor();

  double scale = 1.0;
  for (unsigned int i=0;i<3;i++)
  for (unsigned int j=0;j<3;j++)
  for (unsigned int k=0;k<3;k++)
  for (unsigned int l=0;l<3;l++)
  {
    int voigt1 = voigt_indx(i, j);
    int voigt2 = voigt_indx(k, l);
    int indx = get_array_indx(i, j, k, l);
    double value = tensor[indx];

    if ((voigt1 >= 3) || (voigt2 >= 3))
    {
      value *= 2;
    }
    voigt[voigt1][voigt2] = value;
  }
}

unsigned int EshelbyTensor::voigt_indx(unsigned int i, unsigned int j)
{
  if (i==j)
  {
    return i;
  }

  int max = i > j ? i:j;
  int min = i > j ? j:i;

  if ((min==1) && (max==2)) return 3;
  if ((min==0) && (max==2)) return 4;

  return 5;
}

PyObject* EshelbyTensor::get_raw()
{
  if (require_rebuild) construct_full_tensor();

  PyObject* eshelby_dict = PyDict_New();
  int array[4];
  string key;
  for (unsigned int i=0;i<3;i++)
  for (unsigned int j=0;j<3;j++)
  for (unsigned int k=0;k<3;k++)
  for (unsigned int l=0;l<3;l++)
  {
    array[0] = i;
    array[1] = j;
    array[2] = k;
    array[3] = l;
    array_to_key(key, array);
    int indx = get_array_indx(i, j, k, l);
    PyObject *py_val = PyFloat_FromDouble(tensor[indx]);
    PyDict_SetItemString(eshelby_dict, key.c_str(), py_val);
  }
  return eshelby_dict;
}

void EshelbyTensor::dot(vec6 &voigt)
{
  mat6x6 matrix;
  voigt_representation(matrix);
  vec6 result;
  for (unsigned int i=0;i<6;i++)
  {
    result[i] = 0.0;
    for (unsigned int j=0;j<6;j++)
    {
      result[i] += matrix[i][j]*voigt[j];
    }
  }
  voigt = result;
}
