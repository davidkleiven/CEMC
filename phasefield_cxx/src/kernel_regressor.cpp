#include "kernel_regressor.hpp"
#include "additional_tools.hpp"
#include <sstream>
#include<iostream>
#include <stdexcept>

using namespace std;

const double INF = 1E10;
int KernelRegressor::lower_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }
    double support = kernel->upper() - kernel->lower();
    
    return (x - xmin - support)/kernel_separation();
}

int KernelRegressor::upper_non_zero_kernel(double x) const{
    if (outside_domain(x)){
        return 0;
    }

    double support = kernel->upper() - kernel->lower();
    return (x - xmin + support)/kernel_separation();
}

double KernelRegressor::evaluate(double x) const{
    double value = 0.0;
    for (int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        if ((i >= coeff.size()) || (i < 0)) continue;

        value += coeff[i]*evaluate_kernel(i, x);
    }
    return value;
}

bool KernelRegressor::outside_domain(double x) const{
    return x < xmin || x > xmax;
}

double KernelRegressor::kernel_center(unsigned int i) const{
    return i*kernel_separation() + xmin;
}

double KernelRegressor::deriv(double x) const{
    double value = 0.0;
    for (int i=lower_non_zero_kernel(x);i<=upper_non_zero_kernel(x);i++){
        if ((i >= coeff.size()) || (i < 0)) continue;
        value += coeff[i]*kernel->deriv(x-kernel_center(i));
    }
    return value;
}

double KernelRegressor::kernel_separation() const{
    if (coeff.size() <= 1){
        return INF;
    }

    double domain_size = xmax - xmin;
    double dx = domain_size/(coeff.size() - 1);
    return dx;
}

double KernelRegressor::evaluate_kernel(unsigned int i, double x) const{
    return kernel->evaluate(x-kernel_center(i));
}

PyObject *KernelRegressor::to_dict() const{
    PyObject *dict_repr = PyDict_New();
    PyObject *coeff_list = PyList_New(coeff.size());
    for (unsigned int i=0;i<coeff.size();i++){
        PyObject *pycoeff = PyFloat_FromDouble(coeff[i]);

        // NOTE: SetItem steals the reference from pycoeff
        PyList_SetItem(coeff_list, i, pycoeff);
    }

    PyDict_SetItemString(dict_repr, "coeff", coeff_list);
    PyDict_SetItemString(dict_repr, "xmin", PyFloat_FromDouble(xmin));
    PyDict_SetItemString(dict_repr, "xmax", PyFloat_FromDouble(xmax));
    PyDict_SetItemString(dict_repr, "kernel_name", string2py(kernel->get_name()));
    return dict_repr;
}

void KernelRegressor::from_dict(PyObject *dict_repr){
    string retrieved_name = py2string(PyDict_GetItemString(dict_repr, "kernel_name"));
    if (retrieved_name != kernel->get_name()){
        stringstream ss;
        ss << "Expected that the kernel type is " << kernel->get_name();
        ss << " got " << retrieved_name;
        throw invalid_argument(ss.str());
    }

    // Initialize xmin and xmax
    this->xmin = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "xmin"));
    this->xmax = PyFloat_AsDouble(PyDict_GetItemString(dict_repr, "xmax"));

    // Initialize the coefficients
    PyObject *pycoeff = PyDict_GetItemString(dict_repr, "coeff");
    unsigned int size = PyList_Size(pycoeff);
    coeff.resize(size);
    for (unsigned int i=0;i<size;i++){
        coeff[i] = PyFloat_AsDouble(PyList_GetItem(pycoeff, i));
    }
}