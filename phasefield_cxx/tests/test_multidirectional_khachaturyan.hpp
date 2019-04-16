#ifndef TEST_MULTIKHACHATURYAN_H
#define TEST_MULTIKHACHATURYAN_H

#include <vector>

PyObject* test_functional_derivative(PyObject *elastic, PyObject *misfit, const std::vector<double> &values);

#endif