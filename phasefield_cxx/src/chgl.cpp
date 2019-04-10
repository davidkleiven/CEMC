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
           M(M), alpha(alpha), dt(dt), gl_damping(gl_damping), interface(interface){
               int dims[3] = {L, L, L};
               if (dim == 1){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L);
                    fft = new FFTW(1, dims);
                }
                else if (dim == 2){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L, 0, L);
                    fft = new FFTW(2, dims);
                }
                else if (dim == 3){
                    cmplx_grid_ptr = new MMSP::grid<dim, MMSP::vector<fftw_complex> >(this->num_fields, 0, L, 0, L, 0, L);
                    fft = new FFTW(3, dims);
                }
               check_interface_vector();
           };

template<int dim>
CHGL<dim>::~CHGL(){
    delete cmplx_grid_ptr; cmplx_grid_ptr = nullptr;
    delete fft; fft = nullptr;
}

template<int dim>
void CHGL<dim>::update(int nsteps){

    if (!old_energy_initialized){
        old_energy_initialized = true;
        old_energy = energy();
        cout << "Initial energy: " << old_energy << endl;
    }

    #ifndef HAS_FFTW
        throw runtime_error("CHGL requires FFTW!");
    #endif

    if (!this->free_energy){
        throw runtime_error("Free Energy is not set!");
    }

    from_parent_grid();

    int rank = 0;
	#ifdef MPI_VERSION
    rank = MPI::COMM_WORLD.Get_rank();
    #endif

	MMSP::grid<dim, MMSP::vector<fftw_complex> >& gr = *(this->cmplx_grid_ptr);
	MMSP::ghostswap(gr);

    MMSP::grid<dim, MMSP::vector<fftw_complex> > ft_fields(gr);
    MMSP::grid<dim, MMSP::vector<fftw_complex> > free_energy_real_space(gr);

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

            // Get partial derivative with respect to concentration
            free_eng_deriv[0].re = this->free_energy->partial_deriv_conc(phi_raw_ptr);
            free_eng_deriv[0].im = 0.0;

            for (unsigned int j=1;j<phi.length();j++){
                free_eng_deriv[j].re = this->free_energy->partial_deriv_shape(phi_raw_ptr, j-1);
                free_eng_deriv[j].im = 0.0;
            }
            free_energy_real_space(i) = free_eng_deriv;
        }

        // Fourier transform all the fields --> output in ft_fields
        fft->execute(gr, ft_fields, FFTW_FORWARD, all_fields);
        //save_complex_field("data/chempot.csv", ft_fields, 0);

        // Fourier transform the free energy --> output info grid
        fft->execute(free_energy_real_space, gr, FFTW_FORWARD, all_fields);

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
            ft_fields(i)[0].re = (ft_fields(i)[0].re*(1 + stab_coeff*dt*pow(k, 2)) + gr(i)[0].re*M*dt*pow(k, 2))/(1.0 + 2*M*dt*alpha*pow(k, 4) + dt*stab_coeff*pow(k, 2));

            ft_fields(i)[0].im = (ft_fields(i)[0].im*(1 + stab_coeff*dt*pow(k, 2)) + gr(i)[0].im*M*dt*pow(k, 2))/(1.0 + 2*M*dt*alpha*pow(k, 4) + dt*stab_coeff*pow(k, 2));

            // Update the GL equations
            for (unsigned int field=1;field<MMSP::fields(gr);field++){
                double interface_term = 0.0;
                for (unsigned int dir=0;dir<dim;dir++){
                    interface_term += interface[field-1][dir]*pow(k_vec[dir], 2);
                }

                ft_fields(i)[field].re = (ft_fields(i)[field].re*(1 + stab_coeff*dt*pow(k, 2)) - gr(i)[field].re*gl_damping*dt) / \
                    (1.0 + 2*gl_damping*dt*interface_term + stab_coeff*dt*pow(k, 2));

                ft_fields(i)[field].im = (ft_fields(i)[field].im*(1 + stab_coeff*dt*pow(k, 2)) - gr(i)[field].im*gl_damping*dt) / \
                    (1.0 + 2*gl_damping*dt*interface_term + stab_coeff*dt*pow(k, 2));
            }
        }

        // Inverse Fourier transform --> output intto gr
        fft->execute(ft_fields, gr, FFTW_BACKWARD, all_fields);

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


    double new_energy = energy();

    if ((new_energy > old_energy) && adaptive_dt){
        // We don't transfer the solution
        dt /= 2.0;
        cout << "Timestep refined. New dt = " << dt;
    }
    else{
        // Transfer to parents grid
        old_energy = new_energy;
        to_parent_grid();
    }

    update_counter += 1;

    if ((update_counter%increase_dt == 0) && adaptive_dt){
        dt *= 2.0;
        cout << "Try to increase dt again. New dt = " << dt;
    }

    cout << "Energy: " << new_energy << endl;
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
        (*(this->cmplx_grid_ptr))(i)[field].im = 0.0;
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
        (*(this->cmplx_grid_ptr))(i)[field].im = 0.0;
    }
}

template<int dim>
void CHGL<dim>::print_polynomial() const{
    //cout << *free_energy << endl;
}

template<int dim>
void CHGL<dim>::set_free_energy(const TwoPhaseLandau &poly){
    if (!poly.get_regressor()){
        throw invalid_argument("TwoPhaseLanday has no kernel regressor!");
    }

    if (!poly.get_regressor()->kernel_is_set()){
        throw invalid_argument("The Kernel Regressor has no kernel!");
    }

    if (poly.get_poly_dim() != this->num_fields){
        stringstream ss;
        ss << "The polynomial passed has wrong dimension!";
        ss << "Expected: " << this->num_fields;
        ss << " Got: " << poly.get_poly_dim();
        throw invalid_argument(ss.str());
    }

    free_energy = &poly;
}

template<int dim>
void CHGL<dim>::save_free_energy_map(const std::string &fname) const{
    MMSP::grid<dim, MMSP::vector<double> > free_energy_grid(*this->grid_ptr);

    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++)
    {
        double x[dim+1];
        for (unsigned int field=0;field<MMSP::fields(free_energy_grid);field++){
            x[field] = (*this->grid_ptr)(i)[field];
        }

        free_energy_grid(i)[0] = free_energy->evaluate(x);
        free_energy_grid(i)[1] = free_energy->partial_deriv_conc(x);
        free_energy_grid(i)[2] = free_energy->partial_deriv_shape(x, 0);
    }

    free_energy_grid.output(fname.c_str());
}

template<int dim>
void CHGL<dim>::use_HeLiuTang_stabilizer(double coeff){
    stab_coeff = coeff;

    cout << "Using He-Liu-Tang first order stabilizer with coefficient " << stab_coeff << endl;
};

template<int dim>
double CHGL<dim>::energy() const{

    double integral = 0.0;
    // Calculate the contribution from the free energy
    for (unsigned int i=0;i<MMSP::nodes(*this->grid_ptr);i++){

        // Contribution from free energy
        MMSP::vector<double> phi_real(MMSP::fields(*this->grid_ptr));
        double *phi_raw_ptr = &(phi_real[0]);
        integral += this->free_energy->evaluate(phi_raw_ptr);

        // Contribution from gradient terms
        MMSP::vector<int> pos = this->grid_ptr->position(i);
        MMSP::vector<double> grad = MMSP::gradient(*this->grid_ptr, pos, 0);

        // Add contribution from Cahn-Hilliard
        integral += alpha*pow(norm(grad), 2);

        // Add contribbution from GL fields
        for (unsigned int gl=1;gl < MMSP::fields(*this->grid_ptr);gl++){
            grad = MMSP::gradient(*this->grid_ptr, pos, gl);

            for (unsigned int dir=0;dir<dim;dir++){
                integral += interface[gl-1][dir]*pow(grad[dir], 2);
            }
        }
    }

    return integral/MMSP::nodes(*this->grid_ptr);
}

template<int dim>
void CHGL<dim>::use_adaptive_stepping(double min_dt, unsigned int inc_every){
    adaptive_dt = true;
    minimum_dt=min_dt;
    increase_dt = inc_every;

    cout << "Using adaptive time steps. Min dt: " << min_dt << ". Attempt increase every " << increase_dt << " update.\n";
};

// Explicit instantiations
template class CHGL<1>;
template class CHGL<2>;
template class CHGL<3>;