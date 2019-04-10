#ifndef CHGL_REALSPACE_H
#define CHGL_REALSPACE_H
#include "chgl.hpp"

template<int dim>
class CHGLRealSpace: public CHGL<dim>{
public:
    CHGLRealSpace(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface);
private:
    unsigned int implicitDir{0};
};

#endif