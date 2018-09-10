# distutils: language = c++

from cemc.cpp_ext.wang_landau_sampler cimport WangLandauSampler
cdef class PyWangLandauSampler:
    cdef WangLandauSampler *thisptr

    def __cinit__(self, bc, corr_func, eci, pywl):
        self.thisptr = new WangLandauSampler(bc, corr_func, eci, pywl)

    def __dealloc__(self):
        del self.thisptr

    @property
    def use_inverse_time_algorithm(self):
        return self.thisptr.use_inverse_time_algorithm

    @use_inverse_time_algorithm.setter
    def use_inverse_time_algorithm(self, value):
        self.thisptr.use_inverse_time_algorithm = value

    def use_adaptive_windows(self, min_width):
        self.thisptr.use_adaptive_windows(min_width)

    def save_sub_bin_distribution(self, fname):
        self.thisptr.save_sub_bin_distribution(fname)

    def run(self, maxsteps):
        self.thisptr.run(maxsteps)
