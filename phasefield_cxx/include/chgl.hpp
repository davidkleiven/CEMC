#ifndef CHGL_H
#define CHGL_H
#include "phase_field_simulation.hpp"
#include "polynomial.hpp"
#include "polynomial_term.hpp"
#include <vector>

typedef std::vector<std::vector<double> > interface_vec_t;

template<int dim>
class CHGL: public PhaseFieldSimulation<dim>{
public:
    CHGL(int L, const std::string &prefix, unsigned int num_gl_fields, \
         double M, double alpha, double dt, double gl_damping, 
         const interface_vec_t &interface);

    virtual ~CHGL(){};

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
};
#endif