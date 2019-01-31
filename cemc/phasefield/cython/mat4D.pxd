# distutils: language = c++

cdef extern from "mat4D.hpp":
  cdef cppclass Mat4D:
    Mat4D()

    void from_numpy(object nparray) except +

    object to_numpy()