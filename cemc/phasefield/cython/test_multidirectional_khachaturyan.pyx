cdef extern from "test_multidirectional_khachaturyan.hpp":
    object test_functional_derivative(object elastic, object misfit)


def pytest_functional_derivative(elastic, misfit):
    return test_functional_derivative(elastic, misfit)