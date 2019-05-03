#include "cahn_hilliard_phase_field.hpp"

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

// Explicit instatiations
template class CahnHilliardPhaseField<1>;
template class CahnHilliardPhaseField<2>;
template class CahnHilliardPhaseField<3>;