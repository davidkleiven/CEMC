#ifndef CAHN_HILLIARD_PHASE_FIELD_H
#define CAHN_HILLIARD_PHASE_FIELD_H
#include "phase_field_simulation.hpp"
#include "cahn_hilliard.hpp"

template<int dim>
class CahnHilliardPhaseField: public PhaseFieldSimulation<dim>{
public:
    CahnHilliardPhaseField(int L, \
                         const std::string &prefix, \
                         const CahnHilliard *free_eng, double M, double dt, double alpha);

    virtual ~CahnHilliardPhaseField(){};

    /** Implement the update function */
    virtual void update(int nsteps) override;
    void update_explicit(int nsteps);
    void update_implicit(int nsteps);
private:
    const CahnHilliard *free_eng{nullptr};
    double M;
    double dt;
    double alpha;
    std::string scheme{"explicit"};
};
#endif