#include "regression_kernels.hpp"
#include "additional_tools.hpp"
#include <cmath>

const double PI = acos(-1.0);

PyObject* RegressionKernel::to_dict() const{
    PyObject *dict_repr = PyDict_New();

    PyDict_SetItemString(dict_repr, "lower_limit", PyFloat_FromDouble(lower_limit));
    PyDict_SetItemString(dict_repr, "upper_limit", PyFloat_FromDouble(upper_limit));
    PyDict_SetItemString(dict_repr, "name", string2py(get_name()));
    return dict_repr;
}

void RegressionKernel::from_dict(PyObject *dict_repr){
    lower_limit = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "lower_limit"));
    upper_limit = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "upper_limit"));
    name = py2string(PyDict_GetItemString(dict_repr, "name"));
}

/** Quadratic kernel */
QuadraticKernel::QuadraticKernel(double width): RegressionKernel(), width(width){
    lower_limit = -width;
    upper_limit = width;
    name = "quadratic";
}
double QuadraticKernel::evaluate(double x) const{
    if (is_outside_support(x)){
        return 0.0;
    }

    return amplitude()*(1.0 - x*x/(width*width));
}

bool QuadraticKernel::is_outside_support(double x) const{
    return x < -width || x > width;
}

double QuadraticKernel::deriv(double x) const{
    if (is_outside_support(x)){
        return 0.0;
    }

    return -2*amplitude()*x/(width*width);
}

PyObject *QuadraticKernel::to_dict() const{
    PyObject *dict_repr = RegressionKernel::to_dict();

    PyDict_SetItemString(dict_repr, "width", PyFloat_FromDouble(width));
    return dict_repr;
}

void QuadraticKernel::from_dict(PyObject *dict_repr){
    RegressionKernel::from_dict(dict_repr);
    width = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "width"));
}

/** Gaussian kernel */

GaussianKernel::GaussianKernel(double std_dev): RegressionKernel(), std_dev(std_dev){
    lower_limit = -5*std_dev;
    upper_limit = 5*std_dev;
    name = "gaussian";
};

double GaussianKernel::evaluate(double x) const{
    double prefactor = 1.0/sqrt(2.0*PI*std_dev*std_dev);
    return prefactor*exp(-0.5*pow(x/std_dev, 2));
}

double GaussianKernel::deriv(double x) const{
    return x*evaluate(x)/pow(std_dev, 2);
}

bool GaussianKernel::is_outside_support(double x) const{
    return x < lower_limit || x > upper_limit;
}

PyObject* GaussianKernel::to_dict() const{
    PyObject *dict_repr = RegressionKernel::to_dict();

    PyDict_SetItemString(dict_repr, "std_dev", PyFloat_FromDouble(std_dev));
    return dict_repr;
}

void GaussianKernel::from_dict(PyObject *dict_repr){
    RegressionKernel::from_dict(dict_repr);
    std_dev = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "std_dev"));
}