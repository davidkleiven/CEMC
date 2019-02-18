#include "chgl.hpp"
#include "tools.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

template<int dim>
CHGL<dim>::CHGL(int L, const std::string &prefix, unsigned int num_gl_fields, \
           double M, double alpha, double dt, double gl_damping, 
           const interface_vec_t &interface): PhaseFieldSimulation<dim>(L, prefix, num_gl_fields+1), \
           M(M), alpha(alpha), dt(dt), gl_damping(gl_damping), interface(interface), free_energy(num_gl_fields+1){

               check_interface_vector();
           };


template<int dim>
void CHGL<dim>::add_free_energy_term(double coeff, const PolynomialTerm &polyterm){
    free_energy.add_term(coeff, polyterm);
}

template<int dim>
void CHGL<dim>::update(int nsteps){
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
            MMSP::vector<double> free_eng_deriv(phi.length());

            double *phi_raw_ptr = &(phi[0]);
            for (unsigned int j=0;j<phi.length();j++){
                free_eng_deriv[j] = free_energy.deriv(phi_raw_ptr, j);
            }
            
            // Update the first term (Cahn-Hilliard term)
            temp(i) = free_eng_deriv;
			temp(i)[0] -= 2.0*alpha*lapl_phi[0];

            // Update the Ginzburg-Landau terms
            for (unsigned int dir=0;dir<dim;dir++){
                MMSP::vector<double> lapl_dir = partial_double_derivative(gr, i, dir);
                for (unsigned int gl_eq=0;gl_eq<phi.length()-1;gl_eq++)
                {
                    temp(i)[gl_eq+1] -= 2.0*interface[gl_eq][dir]*lapl_dir[gl_eq+1];
                }
            }
		}

		// MMSP::ghostswap(temp);

		for (int i=0;i<MMSP::nodes(gr);i++){
            // Evaluate the Laplacian for the first field
			double lapl = MMSP::laplacian(temp, i, 0);
			double change = M * lapl;

            // Update according to Cahn-Hilliard
			new_gr(i)[0] = gr(i)[0] + dt*change;

            // Update the Ginzburg-Landau equations
            for (unsigned int gl_eq=0;gl_eq<this->num_fields-1;gl_eq++){
                new_gr(i)[gl_eq+1] = gr(i)[gl_eq+1] - gl_damping*temp(i)[gl_eq+1];
            }
		}
		MMSP::swap(gr, new_gr);
		MMSP::ghostswap(gr);
	}
}

template<int dim>
void CHGL<dim>::check_interface_vector() const{
    if (interface.size() != this->num_fields-1){
        stringstream ss;
        ss << "The number of gradient coefficients does not match ";
        ss << "the number of Ginzburg-Landau equations.";
        ss << "Num. coeff: " << interface.size();
        ss << " Num: GL-equations: " << this->num_fields;
        throw invalid_argument(ss.str());
    }

    // Check that each entry has the correct size
    for (const auto& item : interface){
        if (item.size() != dim){
            stringstream ss;
            ss << "The number of interface terms for each GL equation ";
            ss << "has to match the dimension of the problem.";
            ss << "Dimension: " << dim;
            ss << " Number of interface coefficients " << item.size();
            throw invalid_argument(ss.str());
        }
    }
}


// Explicit instantiations
template class CHGL<1>;
template class CHGL<2>;
template class CHGL<3>;