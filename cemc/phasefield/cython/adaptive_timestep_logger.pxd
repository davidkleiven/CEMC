from libcpp.string cimport string


cdef extern from "adaptive_timestep_logger.hpp":
    cdef struct LogFileEntry:
                unsigned int iter
                double time

    cdef cppclass AdaptiveTimeStepLogger:
        AdaptiveTimeStepLogger(string &fname)

        void log(unsigned int iteration, double time)

        LogFileEntry getLast()

