#include "chgl.hpp"
#include "tools.hpp"
#include <stdexcept>
#include <sstream>
#include <omp.h>
#include <iostream>

using namespace std;

template<int dim>
CHGL<dim>::CHGL(int L, const std::string &prefix, unsigned int num_gl_fields, \
           double M, double alpha, double dt, double gl_damping, 
           const interface_vec_t &interface): PhaseFieldSimulation<dim>(L, prefix, num_gl_fields+1), \
           M(M), alpha(alpha), dt(dt), gl_damping(gl_damping), interface(interface), free_energy(num_gl_fields+1){
               if (dim == 1){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L);
                }
                else if (dim == 2){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L, 0, L);
                }
                else if (dim == 3){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L, 0, L, 0, L);
                }
               check_interface_vector();
           };

template<int dim>
CHGL<dim>::~CHGL(){
    delete cmplx_grid_ptr;
}

template<int dim>
void CHGL<dim>::add_free_energy_term(double coeff, const PolynomialTerm &polyterm){
    free_energy.add_term(coeff, polyterm);
}

template<int dim>
void CHGL<dim>::update(int nsteps){
    #ifndef HAS_FFTW
        throw runtime_error("CHGL requires FFTW!");
    #endif

    if (!is_initialized){
        from_parent_grid();
        is_initialized = true;
    }

    int rank = 0;
	#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	MMSP::grid<dim, MMSP::vector<fftw_complex> >& gr = *(this->cmplx_grid_ptr);
	MMSP::ghostswap(gr);

    MMSP::grid<dim, MMSP::vector<fftw_complex> > free_energy_real_space(gr);
    MMSP::grid<dim, MMSP::vector<fftw_complex> > ft_fields(gr);

	// MMSP::grid<dim, MMSP::vector<fftw_complex> > temp(gr);
	// MMSP::grid<dim, MMSP::vector<fftw_complex> > new_gr(gr);
    int dims[3];
    get_dims(gr, dims);

    vector<int> all_fields;
    for (unsigned int i=0;i<MMSP::fields(gr);i++){
        all_fields.push_back(i);
    }

	for (int step=0;step<nsteps;step++){
		if (rank == 0){
			MMSP::print_progress(step, nsteps);
		}
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
		for (int i=0;i<MMSP::nodes(gr);i++){
			MMSP::vector<fftw_complex> phi = gr(i);
            MMSP::vector<double> phi_real(phi.length());
            for (unsigned int i=0;i<phi.length();i++){
                phi_real[i] = phi[i].re;
            }
            MMSP::vector<fftw_complex> free_eng_deriv(phi.length());
            double *phi_raw_ptr = &(phi_real[0]);
            for (unsigned int j=0;j<phi.length();j++){
                free_eng_deriv[j].re = free_energy.deriv(phi_raw_ptr, j);
                free_eng_deriv[j].im = 0.0;
            }
            free_energy_real_space(i) = free_eng_deriv;
        }

        // Fourier transform all the fields --> output in ft_fields
        fft_mmsp_grid(gr, ft_fields, FFTW_FORWARD, dims, all_fields);

        // Fourier transform the free energy --> output info grid
        fft_mmsp_grid(free_energy_real_space, gr, FFTW_FORWARD, dims, all_fields);

        // Update using semi-implicit scheme
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
		for (int i=0;i<MMSP::nodes(gr);i++){
            MMSP::vector<int> pos = gr.position(i);
            MMSP::vector<double> k_vec(pos.length());
            k_vector(pos, k_vec, this->L);
            double k = norm(k_vec);

            // Update Cahn-Hilliard term
            ft_fields(i)[0].re = (ft_fields(i)[0].re + gr(i)[0].re*M*dt*pow(k, 2))/(1.0 + 2*dt*alpha*pow(k, 4));
            ft_fields(i)[0].im = (ft_fields(i)[0].im + gr(i)[0].im*M*dt*pow(k, 2))/(1.0 + 2*dt*alpha*pow(k, 4));

            // Update the GL equations
            for (unsigned int field=1;field<MMSP::fields(gr);field++){
                double interface_term = 0.0;
                for (unsigned int dir=0;dir<dim;dir++){
                    interface_term += interface[field-1][dir]*pow(k_vec[dir], 2);
                }

                ft_fields(i)[field].re = (ft_fields(i)[field].re - gr(i)[field].re*gl_damping*dt) / \
                    (1.0 - 2*gl_damping*dt*interface_term);

                ft_fields(i)[field].im = (ft_fields(i)[field].im - gr(i)[field].im*gl_damping*dt) / \
                    (1.0 - 2*gl_damping*dt*interface_term);
            }
        }

        // Inverse Fourier transform
        fft_mmsp_grid(ft_fields, gr, FFTW_BACKWARD, dims, all_fields);

        //  MMSP::vector<double> lapl_phi = MMSP::laplacian(gr, i);
        //  MMSP::vector<double> free_eng_deriv(phi.length());
        //     // Update the first term (Cahn-Hilliard term)
        //     temp(i) = free_eng_deriv;
		// 	temp(i)[0] -= 2.0*alpha*lapl_phi[0];

        //     // Update the Ginzburg-Landau terms
        //     for (unsigned int dir=0;dir<dim;dir++){
        //         MMSP::vector<double> lapl_dir = partial_double_derivative(gr, i, dir);
        //         for (unsigned int gl_eq=0;gl_eq<phi.length()-1;gl_eq++)
        //         {
        //             temp(i)[gl_eq+1] -= 2.0*interface[gl_eq][dir]*lapl_dir[gl_eq+1];
        //         }
        //     }
		// }

		// // MMSP::ghostswap(temp);
        // #ifndef NO_PHASEFIELD_PARALLEL
        // #pragma omp parallel for
        // #endif
		// for (int i=0;i<MMSP::nodes(gr);i++){
        //     // Evaluate the Laplacian for the first field
		// 	double lapl = MMSP::laplacian(temp, i, 0);
		// 	double change = M * lapl;

        //     // Update according to Cahn-Hilliard
		// 	new_gr(i)[0] = gr(i)[0] + dt*change;

        //     // Update the Ginzburg-Landau equations
        //     for (unsigned int gl_eq=0;gl_eq<this->num_fields-1;gl_eq++){
        //         new_gr(i)[gl_eq+1] = gr(i)[gl_eq+1] - gl_damping*temp(i)[gl_eq+1];
        //     }
		//}
		//MMSP::swap(gr, new_gr);
		//MMSP::ghostswap(gr);
	}

    // Transfer to parents grid
    to_parent_grid();
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

template<int dim>
void CHGL<dim>::from_parent_grid(){
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++)
    for (unsigned int field=0;field<this->num_fields;field++)
    {
        (*(this->cmplx_grid_ptr))(i)[field].re = (*(this->grid_ptr))(i)[field];
    }
}

template<int dim>
void CHGL<dim>::to_parent_grid() const{
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++)
    for (unsigned int field=0;field<this->num_fields;field++)
    {
        (*(this->grid_ptr))(i)[field] = (*(this->cmplx_grid_ptr))(i)[field].re;
    }
}

template<int dim>
void CHGL<dim>::print_polynomial() const{
    cout << free_energy << endl;
}


// Explicit instantiations
template class CHGL<1>;
template class CHGL<2>;
template class CHGL<3>;