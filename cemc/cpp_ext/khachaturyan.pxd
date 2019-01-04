# distutils: language = c++

cdef extern from "khachaturyan.hpp":
  cdef cppclass Khachaturyan:
    Khachaturyan(object ft_shape, object elastic, object misfit)

    object green_function(object direction) const

    void wave_vector(unsigned int indx[3], double direction[3]) const