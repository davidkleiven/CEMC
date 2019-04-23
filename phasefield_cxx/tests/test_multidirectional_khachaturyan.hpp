#ifndef TEST_MULTIKHACHATURYAN_H
#define TEST_MULTIKHACHATURYAN_H

#include <vector>

PyObject* test_functional_derivative(PyObject *elastic, PyObject *misfit, const std::vector<double> &values);
PyObject *test_contract_tensors(PyObject* tensor1, PyObject *tensor2);
PyObject *test_B_tensor_element(PyObject *pygf, PyObject *tensor1, PyObject *tensor2);


#endif