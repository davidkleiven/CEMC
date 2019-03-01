#ifndef CHGL_H
#define CHGL_H
#include "phase_field_simulation.hpp"
#include "polynomial.hpp"
#include "polynomial_term.hpp"

#ifdef HAS_FFTW
    #include <fftw.h>
#endif
#include "fftw_complex_placeholder.hpp"

#include "MMSP.grid.h"
#include "MMSP.vector.h"
#include <vector>

typedef std::vector<std::vector<double> > interface_vec_t;

template<int dim>
class CHGL: public PhaseFieldSimulation<dim>{
public:
    CHGL(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface);

    virtual ~CHGL();

    /** Add a new free energy term to the model */
    void add_free_energy_term(double coeff, const PolynomialTerm &polyterm);

    /** Implement the update function */
    virtual void update(int nsteps) override;
private:
    double M;
    double alpha;
    double dt;
    double gl_damping;
    interface_vec_t interface;
    Polynomial free_energy;
    MMSP::grid<dim, MMSP::vector<fftw_complex> > *cmplx_grid_ptr{nullptr};

    /** Check that the provided interfaces vector matches requirements */
    void check_interface_vector() const;
    void from_parent_grid();
    void to_parent_grid() const;
    bool is_initialized{false};
};
#endif