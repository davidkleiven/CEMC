#include "chgl_realspace.hpp"
#include "conjugate_gradient.hpp"

template<int dim>
CHGLRealSpace<dim>::CHGLRealSpace(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface): CHGL<dim>(L, prefix, num_gl_fields, M, alpha, dt, gl_damping, interface){};


template<int dim>
void CHGLRealSpace<dim>::build2D(){
    if (dim != 2){
        throw runtime_error("Build2D should never be called when dimension is not 2!");
    }

    did_build_matrices = true;
    MMSP::grid<2, int> indexGrid(1, 0, this->L, 0, this->L);
    for (unsigned int i=0;i<MMSP::nodes(indexGrid);i++){
        indexGrid(i) = i;
    }

    // Build the matrix for the Cahn-Hilliard part
    double factor = 2.0*this->alpha*this->M*this->dt;

    SparseMatrix& matrix = matrices[0];
    matrix.clear();

    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
        matrix.insert(i, i, 1.0 + 20*factor);

        // Retrive node at position +- 1
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-1;j<2;j+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(wrap(pos));
            matrix.insert(i, col, -8*factor);
        }

        // Calculate factor at position +- 2
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-2;j<5;j+=4){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(wrap(pos));
            matrix.insert(i, col, factor);
        }

        // Calculate the cross terms
        for (int ix=-1;ix<2;ix+=2)
        for (int iy=-1;iy<2;iy+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[0] += ix;
            pos[1] += iy;
            unsigned int col = indexGrid(wrap(pos));
            matrix.insert(i, col, 2*factor);
        }
    }

    // Build GL part
    factor = 2*this->gl_damping*this->dt;
    for (unsigned int field=1;field<dim+1;field++){
        SparseMatrix& mat = matrices[field];
        mat.clear();
        for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
            mat.insert(i, i, 1.0 + 2*factor*(this->interface[field-1][0] + this->interface[field-1][1]));

            for (unsigned int dir=0;dir<2;dir++)
            for (int ix=-1;ix<2;ix+=2){
                MMSP::vector<int> pos = this->grid_ptr->position(i);
                pos[dir] += ix;
                unsigned int col = indexGrid(wrap(pos));
                mat.insert(i, col, -factor*this->interface[field-1][dir]);
            }
        }
    }

    matrices[2].save("data/matrixfiel2.csv");
}

template<int dim>
MMSP::vector<int> & CHGLRealSpace<dim>::wrap(MMSP::vector<int> &pos) const{
    for (unsigned int i=0;i<pos.length();i++){
        if (pos[i] < 0){
            pos[i] = this->L - 1;
        }
        else if (pos[i] >= this->L){
            pos[i] = 0;
        }
    }
    return pos;
}


template<int dim>
void CHGLRealSpace<dim>::update(int nsteps){

    if (!did_build_matrices){
        throw runtime_error("The matrices for implicit solution has not been built!");
    }
    int rank = 0;
	#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

    if (!this->old_energy_initialized){
        this->old_energy = energy();
        this->old_energy_initialized = true;
        cout << "Initial energy " << this->old_energy << endl;
    }
    

    // Keep a copy of the grid
    MMSP::grid<dim, MMSP::vector<double> > gr_cpy(*this->grid_ptr);
    gr_cpy.copy(*this->grid_ptr);

    MMSP::grid<dim, MMSP::vector<double> > &gr = *this->grid_ptr;
    MMSP::grid<dim, MMSP::vector<double> > deriv_free_eng(gr);

    ConjugateGradient cg(1E-8);
	MMSP::ghostswap(deriv_free_eng);

    for (int step=0;step<nsteps;step++){
        // Calculate all the derivatives
        if (rank == 0){
                MMSP::print_progress(step, nsteps);
            }
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (int i=0;i<MMSP::nodes(gr);i++){
            MMSP::vector<double> phi = gr(i);
            MMSP::vector<double> free_eng_deriv(phi.length());

            double *phi_raw_ptr = &(phi[0]);

            // Get partial derivative with respect to concentration
            free_eng_deriv[0]= this->free_energy->partial_deriv_conc(phi_raw_ptr);

            for (unsigned int j=1;j<phi.length();j++){
                free_eng_deriv[j] = this->free_energy->partial_deriv_shape(phi_raw_ptr, j-1);
            }
            deriv_free_eng(i) = free_eng_deriv;
        }

        // Solve each field with the conjugate gradient method
        for (unsigned int field=0;field<MMSP::fields(gr);field++){
            vector<double> rhs;
            vector<double> field_values;
            for (unsigned int i=0;i<MMSP::nodes(gr);i++){
                field_values.push_back(gr(i)[field]);

                if (field == 0){
                    rhs.push_back(gr(i)[field] + this->dt*this->M*MMSP::laplacian(deriv_free_eng, i, 0));
                }
                else{
                    rhs.push_back(gr(i)[field] - this->dt*this->gl_damping*deriv_free_eng(i)[field]);
                }
            }

            // Solve with CG
            cg.solve(matrices[field], rhs, field_values);

            // Transfer the field values back
            for (unsigned int i=0;i<MMSP::nodes(gr);i++){
                gr(i)[field] = field_values[i];
            }
        }
    }

    // Calculate the energy
    double new_energy = energy();
    bool did_lower_timestep = false;
    if (((new_energy > this->old_energy) && (this->adaptive_dt)) || isnan(new_energy)){
        this->dt /= 2.0; // Reduce time step
        build2D(); // Rebuild matrices
        gr_cpy.swap(*this->grid_ptr);
        did_lower_timestep = true;

        cout << "Refine timestep. New dt: " << this->dt  << ". Energy obtained: " << new_energy << endl;
    }
    else{
        cout << "Energy: " << new_energy << endl;
        this->old_energy = new_energy;
    }

    this->update_counter += 1;

    if ((this->update_counter%this->increase_dt == 0) && this->adaptive_dt && !did_lower_timestep){
        this->dt *= 2;
        build2D();
        cout << "Time step increased. New dt: " << this->dt << endl;
    }

    if (this->dt < this->minimum_dt){
        cout << "Reached minimum timestep\n";
        this->quit = true;
    }
}

template<int dim>
double CHGLRealSpace<dim>::energy() const{

    double integral = 0.0;

    // Construct a temperatry copy
    MMSP::grid<dim, MMSP::vector<double> >& gr = *this->grid_ptr;

    // Calculate the contribution from the free energy
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(+ : integral)
    #endif
    for (unsigned int i=0;i<MMSP::nodes(gr);i++){

        // Contribution from free energy
        MMSP::vector<double> phi_real(MMSP::fields(gr));
        double *phi_raw_ptr = &(phi_real[0]);
        integral += this->free_energy->evaluate(phi_raw_ptr);

        // Contribution from gradient terms
        MMSP::vector<int> pos = gr.position(i);
        MMSP::vector<double> grad = MMSP::gradient(gr, pos, 0);

        // Add contribution from Cahn-Hilliard
        integral += this->alpha*pow(norm(grad), 2);

        // Add contribbution from GL fields
        for (unsigned int gl=1;gl < MMSP::fields(gr);gl++){
            grad = MMSP::gradient(gr, pos, gl);

            for (unsigned int dir=0;dir<dim;dir++){
                integral += this->interface[gl-1][dir]*pow(grad[dir], 2);
            }
        }
    }

    return integral/MMSP::nodes(gr);
}    

template<int dim>
double CHGLRealSpace<dim>::laplacian_central_stencil() const{
    switch(dim){
        case 1:
            return -2.0;
        case 2:
            return -4.0;
        case 3:
            return -6.0;
        default:
            throw invalid_argument("Dimension has to be 1, 2 or 3!");
    }
}

// Explicit instantiations
template class CHGLRealSpace<1>;
template class CHGLRealSpace<2>;
template class CHGLRealSpace<3>;
