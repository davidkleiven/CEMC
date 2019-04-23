from libcpp.vector cimport vector

cdef extern from "test_multidirectional_khachaturyan.hpp":
    object test_functional_derivative(object elastic, object misfit, const vector[double] &values) except+
    object test_contract_tensors(object tensor1, object tensor2)
    object test_B_tensor_element(vector[double] direction, object gf, object t1, object t2)


def pytest_functional_derivative(elastic, misfit, field):
    cdef vector[double] c_vec

    for i in range(len(field)):
        c_vec.push_back(field[i])

    return test_functional_derivative(elastic, misfit, c_vec)

def pytest_contract_tensors(t1, t2):
    return test_contract_tensors(t1, t2)

def pytest_B_tensor_element(dir, gf, t1, t2):
    cdef vector[double] v
    for i in range(3):
        v.push_back(dir[i])
    return test_B_tensor_element(v, gf, t1, t2)