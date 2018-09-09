# distutils: language = c++

from cemc.cpp_ext.eshelby_tensor cimport EshelbyTensor, EshelbySphere
from cython.operator import dereference as deref

cdef class PyEshelbyTensor:
    cdef EshelbyTensor *eshelby_cpp

    def __cinit__(PyEshelbyTensor self, double a, double b, double c,
                  double poisson):
        self.eshelby_cpp = new EshelbyTensor(a, b, c, poisson)

    def __dealloc__(self):
        if type(self) is PyEshelbyTensor:
            del self.eshelby_cpp

    def __call__(self, i1, i2, i3, i4):
        return deref(self.eshelby_cpp)(i1, i2, i3, i4)

    def get_raw(self):
        return self.eshelby_cpp.get_raw()

    def aslist(self):
        return self.eshelby_cpp.aslist()


cdef class PyEshelbySphere(PyEshelbyTensor):
    cdef EshelbySphere *derived_ptr

    def __cinit__(PyEshelbySphere self, double a, double b_dummy,
                  double c_dummy, double poisson):
        self.derived_ptr = self.eshelby_cpp = new EshelbySphere(a, poisson)

    def __dealloc__(self):
        if type(self) is PyEshelbySphere:
            del self.derived_ptr
