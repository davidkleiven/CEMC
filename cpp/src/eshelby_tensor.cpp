#include "eshelby_tensor.hpp"
#include <Python.h>
#include <cmath>
#include <stdexcept>

using namespace std;

const double PI = acos(-1.0);
EshelbyTensor::EshelbyTensor(double a, double b, double c, double poisson):a(a), b(b), c(c), poisson(poisson)
{
  if ((a < b) || (a < c) || (b<c))
  {
    throw invalid_argument("The Eshelby tensor assumes that a > b > c!");
  }

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
