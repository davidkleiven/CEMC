#ifndef CAHN_HILLIARD_PHASE_FIELD_H
#define CAHN_HILLIARD_PHASE_FIELD_H
#include "phase_field_simulation.hpp"
#include "cahn_hilliard.hpp"
#include "sparse_matrix.hpp"

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
    void build2D();
    void set_adaptive(double min_dt, double max_change);
private:
    const CahnHilliard *free_eng{nullptr};
    double M;
    double dt;
    double alpha;
    bool did_build_matrix{false};
    double min_dt{0.0};
    double max_change{1.0};
    std::string scheme{"explicit"};
    SparseMatrix system_matrix;
};
#endif