# distutils: language = c++

cdef extern from "init_numpy.hpp":
  pass

cdef extern from "eshelby_tensor.hpp":
    cdef cppclass EshelbyTensor:
        EshelbyTensor(double a, double b, double c, double poisson)

        object aslist()

        object get_raw()

        double operator()(int i1, int i2, int i3, int i4)

cdef extern from "eshelby_sphere.hpp":
    cdef cppclass EshelbySphere(EshelbyTensor):
        EshelbySphere(double a, double poisson)
