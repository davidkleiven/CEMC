# distutils: language = c++

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "wang_landau_sampler.hpp":
    cdef cppclass WangLandauSampler:
        bool use_inverse_time_algorithm

        WangLandauSampler(object atoms, object BC, object corrFunc, object ecis, object py_wl )

        void use_adaptive_windows(unsigned int min_window_width)

        void run(unsigned int maxsteps)

        void save_sub_bin_distribution(string fname)
