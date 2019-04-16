from libcpp.vector cimport vector

cdef extern from "test_multidirectional_khachaturyan.hpp":
    object test_functional_derivative(object elastic, object misfit, const vector[double] &values) except+


def pytest_functional_derivative(elastic, misfit, field):
    cdef vector[double] c_vec

    for i in range(len(field)):
        c_vec.push_back(field[i])

    return test_functional_derivative(elastic, misfit, c_vec)