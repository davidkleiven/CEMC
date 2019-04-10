#include "chgl_realspace.hpp"

template<int dim>
CHGLRealSpace<dim>::CHGLRealSpace(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface): CHGL<dim>(L, prefix, num_gl_fields, M, alpha, dt, gl_damping, interface){};


template<int dim>
void CHGLRealSpace<dim>::build2D(){
    if (dim != 2){
        throw runtime_error("Build2D should never be called when dimension is not 2!");
    }

    MMSP::grid<2, int> indexGrid(1, 0, this->L, 0, this->L);
    for (unsigned int i=0;i<MMSP::nodes(indexGrid);i++){
        indexGrid(i) = i;
    }

    // Build the matrix for the Cahn-Hilliard part
    double factor = 2.0*this->alpha*this->M*this->dt;

    SparseMatrix& matrix = matrices[0];
    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
        matrix.insert(i, i, 1.0 + 20*factor);

        // Retrive node at position +- 1
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-1;j<2;j+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(pos);
            matrix.insert(i, col, -8*factor);
        }

        // Calculate factor at position +- 2
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-2;j<5;j+=4){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(pos);
            matrix.insert(i, col, factor);
        }

        // Calculate the cross terms
        for (int ix=-1;ix<2;ix+=2)
        for (int iy=-1;iy<2;iy+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[0] += ix;
            pos[1] += iy;
            unsigned int col = indexGrid(pos);
            matrix.insert(i, col, 2*factor);
        }
    }

    // Build GL part
    factor = 2*this->L*this->dt;
    for (unsigned int field=1;field<dim+1;field++){
        SparseMatrix& mat = matrices[field];
        for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
            matrix.insert(i, i, 1.0 + 2*factor*(this->interface[field-1][0] + this->interface[field-1][1]));

            for (unsigned int dir=0;dir<2;dir++)
            for (int ix=-1;ix<2;ix+=2){
                MMSP::vector<int> pos = this->grid_ptr->position(i);
                pos[dir] += ix;
                unsigned int col = indexGrid(pos);
                matrix.insert(i, col, -factor*this->interface[field-1][dir]);
            }
        }
    }
}


// Explicit instantiations
template class CHGLRealSpace<1>;
template class CHGLRealSpace<2>;
template class CHGLRealSpace<3>;
