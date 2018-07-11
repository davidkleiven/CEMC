#include "eshelby_tensor.hpp"
#include <Python.h>
#include <cmath>
#include <stdexcept>
#include <map>
#include <string>

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
  I_tensor(I);
}

double EshelbyTensor::elliptic_integral_python(double theta, double kappa, const char* funcname)
{
  PyObject* mod_string = PyString_FromString("scipy.special");
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

double EshelbyTensor::F(double theta, double kappa)
{
  return elliptic_integral_python(theta, kappa, "ellipkinc");
}

double EshelbyTensor::E(double theta, double kappa)
{
  return elliptic_integral_python(theta, kappa, "ellipeinc");
}


void EshelbyTensor::I_tensor(array<double,6> &result) const
{
  double f = elliptic_f;
  double e = elliptic_e;

  double I1 = 4.0*PI*a*b*c*(f-e)/( (a*a-b*b)*sqrt(a*a-c*c) );
  double f_factor = b*sqrt(a*a-c*c)/(a*c);
  double I3 = 4.0*PI*a*b*c*(f_factor-e)/( (b*b-c*c)*sqrt(a*a-c*c) );

  double I2 = 4.0*PI - I1 - I3;
  double I12 = (I2-I1)/(a*a-b*b);

  double I11 = 3.0*I1 - 4.0*PI*c*c/pow(a,2) - (b*b-c*c)*I12;
  I11 /= (3.0*(a*a-c*c));
  double I13 = 4.0*PI/(a*a) - 3.0*I11 - I12;

  result[0] = I1;
  result[1] = I2;
  result[2] = I3;
  result[3] = I11;
  result[4] = I12;
  result[5] = I13;
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
  double semi_axes[3] = {a, b, c};
  for (unsigned int i=0;i<81;i++)
  {
    tensor[i] = 0.0;
  }

  for (unsigned int i=0;i<3;i++ )
  {
    map<string, double> ref_tensor;
    construct_ref_tensor(ref_tensor, semi_axes);

    for (auto iter=ref_tensor.begin(); iter != ref_tensor.end(); ++iter)
    {
      int indices[4];
      key_to_array(iter->first, indices);

      // Update the array to the current cyclic permutation
      for (unsigned int n=0;n<4;n++)
      {
        indices[n] = (indices[n]+1)%3;
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
    circular_shift(semi_axes, 3);
  }
  require_rebuild = false;
}

void EshelbyTensor::construct_ref_tensor(map<string, double> &elements, double semi_axes[3])
{
  double a_cyc = semi_axes[0];
  double b_cyc = semi_axes[1];
  double c_cyc = semi_axes[2];

  double theta = asin(sqrt(a_cyc*a_cyc - c_cyc*c_cyc)/a_cyc*a_cyc);
  double kappa;
  if (abs(a_cyc*a_cyc - c_cyc*c_cyc) < 1E-6)
  {
    kappa = 0.0;
  }
  else
  {
    kappa = sqrt((a_cyc*a_cyc - b_cyc*b_cyc) / (a_cyc*a_cyc - c_cyc*c_cyc));
  }

  double f_int = F(theta, kappa);
  double e_int = E(theta, kappa);

  double I1 = 4.0*PI*a_cyc*b_cyc*c_cyc / sqrt((a_cyc*a_cyc - b_cyc*b_cyc) * (a_cyc*a_cyc - c_cyc*c_cyc));
  I1 *= (f_int - e_int);

  double I3 = 4.0*PI*a_cyc*b_cyc*c_cyc / sqrt((b_cyc*b_cyc - c_cyc*c_cyc) * (a_cyc*a_cyc - b_cyc*b_cyc));
  I3 *= ( (b_cyc*sqrt(a_cyc*a_cyc - c_cyc*c_cyc)) / (a_cyc*c_cyc) - e_int);

  double I2 = 4.0*PI - I1 - I3;

  double I12 = (I2 - I1) / (a_cyc*a_cyc - b_cyc*b_cyc);
  double I11 = 4.0*PI/(a_cyc*a_cyc) - 3*I1/(c_cyc*c_cyc) + (b_cyc*b_cyc/(c_cyc*c_cyc) - 1.0)*I2;
  I11 /= (3.0*(1.0 - a_cyc*a_cyc/(c_cyc*c_cyc)));
  double I13 = 3*I1/(c_cyc*c_cyc) - 3*a_cyc*a_cyc*I11/(c_cyc*c_cyc) - b_cyc*b_cyc*I12/(c_cyc*c_cyc);

  double pref = 1.0/(8.0*PI*(1-poisson));

  // S_1111
  elements["0000"] = pref*(3.0*a_cyc*a_cyc*I11 + (1.0-2*poisson)*I1);

  // S_1122
  elements["0011"] = pref*(b_cyc*b_cyc*I12 + (1.0-2.0*poisson)*I1);

  // S_1133
  elements["0022"] = pref*(c_cyc*c_cyc*I13 + (1.0-2.0*poisson)*I1);

  // S_1212
  elements["0101"] = 0.5*pref*( (a_cyc*a_cyc + b_cyc*b_cyc)*I12 - (1.0-2.0*poisson)*(I1+I2));

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

void EshelbyTensor::circular_shift(double data[], int size)
{
  double first = data[0];
  for (unsigned int i=0;i<size-1;i++)
  {
    data[i] = data[i+1];
  }
  data[size-1] = first;
}
