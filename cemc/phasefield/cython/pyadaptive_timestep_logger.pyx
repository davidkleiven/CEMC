#cimport cemc.phasefield.cython.adaptive_timestep_logger as adaptive_timestep_logger
#from adaptive_timestep_logger cimport AdaptiveTimeStepLogger, LogFileEntry
from cemc.phasefield.cython.adaptive_timestep_logger cimport AdaptiveTimeStepLogger, LogFileEntry

cdef class PyAdaptiveTimeStepLogger:
    cdef AdaptiveTimeStepLogger *thisptr

    def __cinit__(self, fname):
        self.thisptr = new AdaptiveTimeStepLogger(fname)

    def __dealloc__(self):
        del self.thisptr

    def log(self, iter, time):
        self.thisptr.log(iter, time)

    def getLast(self):
        cdef LogFileEntry entry = self.thisptr.getLast()
        return {"time": entry.time, "iter": entry.iter}