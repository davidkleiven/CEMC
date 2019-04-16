# distutils: language = c++

cdef extern from "khachaturyan.hpp":
  cdef cppclass Khachaturyan:
    Khachaturyan(unsigned int dim, object elastic, object misfit)

    object green_function(object direction) const

    void wave_vector(unsigned int indx[3], double direction[3], int N) const

    double zeroth_order_integral(object ft_shp)

    void effective_stress(double eff_stress[3][3]) const