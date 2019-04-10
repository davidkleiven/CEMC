#ifndef CHGL_REALSPACE_H
#define CHGL_REALSPACE_H
#include "chgl.hpp"
#include "sparse_matrix.hpp"
#include <array>

template<int dim>
class CHGLRealSpace: public CHGL<dim>{
public:
    CHGLRealSpace(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface);

    /** Build the matrix for CHGL equations */
    void build2D();
private:
    unsigned int implicitDir{0};
    std::array<SparseMatrix, dim+1> matrices;

    unsigned int node_index(const MMSP::vector<int> &pos) const;
};

#endif