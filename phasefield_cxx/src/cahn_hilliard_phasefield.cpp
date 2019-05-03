#include "cahn_hilliard_phase_field.hpp"
#include "tools.hpp"
#include "conjugate_gradient.hpp"

template<int dim>
CahnHilliardPhaseField<dim>::CahnHilliardPhaseField(int L, \
                         const std::string &prefix,\
                         const CahnHilliard *free_eng, double M, double dt, double alpha): \
						 PhaseFieldSimulation<dim>(L, prefix, 1), free_eng(free_eng), M(M), dt(dt), alpha(alpha){};


template<int dim>
void CahnHilliardPhaseField<dim>::update(int nsteps){
	if (scheme == "explicit"){
		update_explicit(nsteps);
	}
	else if (scheme == "implicit"){
		update_implicit(nsteps);
	}
	else{
		throw invalid_argument("Unknown update scheme!");
	}
}

template<int dim>
void CahnHilliardPhaseField<dim>::update_explicit(int nsteps){
	int rank = 0;
	#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	MMSP::grid<dim, MMSP::vector<double> >& gr = *this->grid_ptr;
	MMSP::ghostswap(gr);

	MMSP::grid<dim, MMSP::vector<double> > temp(gr);
	MMSP::grid<dim, MMSP::vector<double> > new_gr(gr);

	for (int step=0;step<nsteps;step++){
		if (rank == 0){
			MMSP::print_progress(step, nsteps);
		}

		for (int i=0;i<MMSP::nodes(gr);i++){
			MMSP::vector<double> phi = gr(i);
			MMSP::vector<double> lapl_phi = MMSP::laplacian(gr, i);

			double free_eng_deriv = free_eng->deriv(phi[0]);
			double new_value = free_eng_deriv - 2.0*alpha*lapl_phi[0];
			temp(i)[0] = new_value;
		}

		// MMSP::ghostswap(temp);

		for (int i=0;i<MMSP::nodes(gr);i++){
			MMSP::vector<double> lapl = MMSP::laplacian(temp, i);
			double change = M * lapl[0];
			new_gr(i)[0] = gr(i)[0] + dt*change;
		}
		MMSP::swap(gr, new_gr);
		MMSP::ghostswap(gr);
	}
}

template<int dim>
void CahnHilliardPhaseField<dim>::build2D(){
    if (dim != 2){
        throw runtime_error("Build2D should never be called when dimension is not 2!");
    }

    did_build_matrix = true;
    MMSP::grid<2, int> indexGrid(1, 0, this->L, 0, this->L);
    for (unsigned int i=0;i<MMSP::nodes(indexGrid);i++){
        indexGrid(i) = i;
    }

    // Build the matrix for the Cahn-Hilliard part
    double factor = 2.0*this->alpha*this->M*this->dt;

    system_matrix.clear();

    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
        system_matrix.insert(i, i, 1.0 + 20*factor);

        // Retrive node at position +- 1
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-1;j<2;j+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(wrap(pos, this->L));
            system_matrix.insert(i, col, -8*factor);
        }

        // Calculate factor at position +- 2
        for (unsigned int dir=0;dir<2;dir++)
        for (int j=-2;j<5;j+=4){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[dir] += j;
            unsigned int col = indexGrid(wrap(pos, this->L));
            system_matrix.insert(i, col, factor);
        }

        // Calculate the cross terms
        for (int ix=-1;ix<2;ix+=2)
        for (int iy=-1;iy<2;iy+=2){
            MMSP::vector<int> pos = this->grid_ptr->position(i);
            pos[0] += ix;
            pos[1] += iy;
            unsigned int col = indexGrid(wrap(pos, this->L));
            system_matrix.insert(i, col, 2*factor);
        }
    }

    // Sanity check: All matrices should be symmetric
    unsigned int counter = 0;
	if (!system_matrix.is_symmetric()){
		stringstream ss;
		ss << "System matrix is not symmetric!";
		throw runtime_error(ss.str());
	}

	system_matrix.to_csr();
}

template<int dim>
void CahnHilliardPhaseField<dim>::update_implicit(int nsteps){
	if (!did_build_matrix){
        throw runtime_error("The matrices for implicit solution has not been built!");
    }

	// Keep a copy of the grid
	MMSP::grid<dim, MMSP::vector<double> > &gr = *this->grid_ptr;
    MMSP::grid<dim, MMSP::vector<double> > gr_cpy(*this->grid_ptr);
	MMSP::grid<dim, double > deriv_free_eng(gr, 1);
    gr_cpy.copy(*this->grid_ptr);

	ConjugateGradient cg(1E-8);

	int rank = 0;
	#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif
	for (int step=0;step<nsteps;step++){
		if (rank == 0){
			MMSP::print_progress(step, nsteps);
		}

		#ifndef NO_PHASEFIELD_PARALLEL
		#pragma omp parallel for
		#endif
		for (int i=0;i<MMSP::nodes(gr);i++){
			MMSP::vector<double> phi = gr(i);
			MMSP::vector<double> lapl_phi = MMSP::laplacian(gr, i);

			deriv_free_eng(i) = free_eng->deriv(phi[0]);
		}

		vector<double> rhs(MMSP::nodes(gr));
        vector<double> field_values(MMSP::nodes(gr));

		#ifndef NO_PHASEFIELD_PARALLEL
		#pragma omp parallel for
		#endif
		for (int i=0;i<MMSP::nodes(gr);i++){
			rhs[i] = gr(i)[0] + this->dt*this->M*MMSP::laplacian(deriv_free_eng, i);
			field_values[i] = gr(i)[0];
		}
		
		cg.solve(system_matrix, rhs, field_values);

		// Transfer the field values back
		#ifndef NO_PHASEFIELD_PARALLEL
		#pragma omp parallel for
		#endif
		for (unsigned int i=0;i<MMSP::nodes(gr);i++){
			gr(i)[0] = field_values[i];
		}

		double max_diff = inf_norm_diff(gr, gr_cpy);

		if (max_diff > this->max_change){
			this->dt /= 2.0;
			build2D();
			this->grid_ptr->copy(gr_cpy);
            cout << "Refine timestep. New dt: " << this->dt << ". Max change: " << max_diff << endl;
		}
		else{
			gr_cpy.copy(*this->grid_ptr);
		}
	}

	if (this->dt < this->min_dt){
		cout << "Reached minimum timestep\n";
		this->quit = true;
	}

	this->dt *= 2;
	this->build2D();
	cout << "New time step: " << this->dt << endl;
}

template<int dim>
void CahnHilliardPhaseField<dim>::set_adaptive(double min_dt, double max_change){
	this->min_dt = min_dt;
	this->max_change = max_change;
	this->scheme = "implicit";
}


// Explicit instatiations
template class CahnHilliardPhaseField<1>;
template class CahnHilliardPhaseField<2>;
template class CahnHilliardPhaseField<3>;