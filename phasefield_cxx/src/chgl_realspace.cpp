#include "chgl_realspace.hpp"
#include "conjugate_gradient.hpp"
#include <limits>

template<int dim>
CHGLRealSpace<dim>::CHGLRealSpace(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface): CHGL<dim>(L, prefix, num_gl_fields, M, alpha, dt, gl_damping, interface){
             strain_deriv = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(*this->grid_ptr, MMSP::fields(*this->grid_ptr));
         };

template<int dim>
CHGLRealSpace<dim>::~CHGLRealSpace(){
    delete strain_deriv; strain_deriv = nullptr;
}

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

    // Sanity check: All matrices should be symmetric
    unsigned int counter = 0;
    for (const auto& mat : matrices){
        if (!mat.is_symmetric()){
            stringstream ss;
            ss << "Mass matrix " << counter << " is not symmetric!";
            throw runtime_error(ss.str());
        }
        counter++;
    }
}

template<int dim>
MMSP::vector<int> & CHGLRealSpace<dim>::wrap(MMSP::vector<int> &pos) const{
    for (unsigned int i=0;i<pos.length();i++){
        if (pos[i] < 0){
            pos[i] += this->L;
        }
        else if (pos[i] >= this->L){
            pos[i] -= this->L;
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

    // Keep a copy of the grid
    MMSP::grid<dim, MMSP::vector<double> > gr_cpy(*this->grid_ptr);
    gr_cpy.copy(*this->grid_ptr);

    MMSP::grid<dim, MMSP::vector<double> > &gr = *this->grid_ptr;
    MMSP::grid<dim, MMSP::vector<double> > deriv_free_eng(gr);

    ConjugateGradient cg(1E-8);
	MMSP::ghostswap(deriv_free_eng);

    bool did_lower_timestep = false;
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

        // Calculate the strain functional derivatives
        calculate_strain_contribution();

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

            // Add Cook noise (note this function does not do anything if the 
            // user has not requested to include noise)
            add_cook_noise_to_fd_scheme(rhs, field);

            if (field > 0){
                add_strain_contribution(rhs, field);
            }

            // Solve with CG
            cg.solve(matrices[field], rhs, field_values);

            // Transfer the field values back
            for (unsigned int i=0;i<MMSP::nodes(gr);i++){
                gr(i)[field] = field_values[i];
            }
        }

        double max_diff = inf_norm_diff(gr, gr_cpy);

        if (should_lower_timestep(max_diff)){
            this->set_timestep(this->dt/2.0);// Reduce time step
            build2D(); // Rebuild matrices
            //gr_cpy.swap(*this->grid_ptr);
            this->grid_ptr->copy(gr_cpy);
            did_lower_timestep = true;
            cout << "Refine timestep. New dt: " << this->dt << ". Max change: " << max_diff << endl;
        }
        else{
            gr_cpy.copy(*this->grid_ptr);
        }
    }

    // Calculate the energy
    

    this->update_counter += 1;

    if ((this->update_counter%this->increase_dt == 0) && this->adaptive_dt && !did_lower_timestep){
        this->set_timestep(this->dt*2);
        build2D();
        cout << "Time step increased. New dt: " << this->dt << endl;
    }

    if (this->dt < this->minimum_dt){
        cout << "Reached minimum timestep\n";
        this->quit = true;
    }

    map<string, double> energy_values;
    energy(energy_values);
    
    if (this->khachaturyan.num_models() > 0){
        energy_values["strain_energy"] = this->khachaturyan.get_last_strain_energy();
        energy_values["min_shape_func_deriv"] = min_strain_deriv;
        energy_values["max_shape_func_deriv"] = max_strain_deriv;
    }

    log_tritem(energy_values);
    this->track_values.push_back(energy_values);
}

template<int dim>
void CHGLRealSpace<dim>::energy(std::map<std::string, double> &tr_item) const{

    double integral = 0.0;
    double surf_integral = 0.0;

    // Construct a temperatry copy
    MMSP::grid<dim, MMSP::vector<double> >& gr = *this->grid_ptr;

    // Calculate the contribution from the free energy
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(+ : integral, surf_integral)
    #endif
    for (unsigned int i=0;i<MMSP::nodes(gr);i++){

        // Contribution from free energy
        //MMSP::vector<double> phi_real(MMSP::fields(gr));
        MMSP::vector<double> phi_real = gr(i);
        double *phi_raw_ptr = &(phi_real[0]);
        integral += this->free_energy->evaluate(phi_raw_ptr);

        // Contribution from gradient terms
        MMSP::vector<int> pos = gr.position(i);
        MMSP::vector<double> grad = MMSP::gradient(gr, pos, 0);

        // Add contribution from Cahn-Hilliard
        surf_integral += this->alpha*pow(norm(grad), 2);

        // Add contribbution from GL fields
        for (unsigned int gl=1;gl < MMSP::fields(gr);gl++){
            grad = MMSP::gradient(gr, pos, gl);

            for (unsigned int dir=0;dir<dim;dir++){
                surf_integral += this->interface[gl-1][dir]*pow(grad[dir], 2);
            }
        }
    }

    tr_item["vol_energy"] = integral/MMSP::nodes(gr);
    tr_item["surf_energy"] = surf_integral/MMSP::nodes(gr);
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

template<int dim>
void CHGLRealSpace<dim>::add_cook_noise_to_fd_scheme(std::vector<double> &rhs, int field) const{
    if (this->cook_noise.size() == 0){
        return;
    }

    vector<double> noise(rhs.size());
    this->cook_noise[field]->create(noise);

    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<rhs.size();i++){
        rhs[i] += this->dt*noise[i];
    }
}   

template<int dim>
bool CHGLRealSpace<dim>::should_lower_timestep(double diff) const{
    if (!this->adaptive_dt){
        return false;
    }

    if (diff < this->lower_energy_cut){
        return false;
    }

    return true;
}

template<int dim>
void CHGLRealSpace<dim>::calculate_strain_contribution(){
    if (this->khachaturyan.num_models() == 0){
        return;
    }

    vector<int> shape_fields;
    for (unsigned int i=0;i<dim;i++){
        shape_fields.push_back(i+1);
    }

    // Create one complex array with the current fields
    MMSP::grid<dim, MMSP::vector<fftw_complex> > grid_in(*this->strain_deriv);
    

    // Transfer data grid in
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<MMSP::nodes(grid_in);i++)
    for (unsigned int gl_field=0;gl_field<dim;gl_field++){
        real(grid_in(i)[gl_field]) = (*this->grid_ptr)(i)[gl_field+1];
        imag(grid_in(i)[gl_field]) = 0.0;
    }

    this->khachaturyan.functional_derivative(grid_in, *strain_deriv, shape_fields);
    update_min_max_strain_deriv();
}

template<int dim>
void CHGLRealSpace<dim>::add_strain_contribution(std::vector<double> &rhs, int field) const{
    if (this->khachaturyan.num_models() == 0){
        return;
    }

    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<rhs.size();i++){
        rhs[i] -= this->dt*this->gl_damping*real((*strain_deriv)(i)[field]);
    }
}

template<int dim>
void CHGLRealSpace<dim>::log_tritem(const map<string, double> &item) const{
    for (auto iter=item.begin(); iter != item.end(); ++iter){
        cout << iter->first << ": " << iter->second << " ";
    }
    cout << endl;
}

template<int dim>
void CHGLRealSpace<dim>::update_min_max_strain_deriv(){
    min_strain_deriv = numeric_limits<double>::max();
    max_strain_deriv = numeric_limits<double>::min();;
    unsigned int num_nan = 0;

    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(min: min_strain_deriv) reduction(max: max_strain_deriv) reduction(+: num_nan)
    #endif
    for (unsigned int i=0;i<MMSP::nodes(*strain_deriv);i++){
        for (unsigned int field=1;field<dim+1;field++){

            double value = real((*strain_deriv)(i)[field]);

            if (isnan(value)){
                num_nan += 1;
            }

            if (value < min_strain_deriv){
                min_strain_deriv = value;
            }
            else if (value > max_strain_deriv){
                max_strain_deriv = value;
            }
        }
    }

    if (num_nan > 0){
        stringstream ss;
        ss << num_nan << " functional derivatives is NaN!";
        throw runtime_error(ss.str());
    }
}
// Explicit instantiations
template class CHGLRealSpace<1>;
template class CHGLRealSpace<2>;
template class CHGLRealSpace<3>;
