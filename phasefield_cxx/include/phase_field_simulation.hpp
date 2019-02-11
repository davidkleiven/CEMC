#ifndef PHASE_FIELD_SIMULATION_H
#define PHASE_FIELD_SIMULATION_H
//#include "MMSP.grid.h"
//#include "MMSP.vector.h"
#include "MMSP.grid.h"
#include "MMSP.vector.h"

#include <string>
#include <vector>

template<int dim>
class PhaseFieldSimulation{
public:
    PhaseFieldSimulation(int L, \
                         const std::string &prefix, unsigned int num_fields);
    virtual ~PhaseFieldSimulation();

    /** Initialize a given field with random numbers */
    void random_initialization(unsigned int field_no, double lower, double upper);
    void random_initialization(double lower, double upper);

    /** Update function */
    virtual void update(int steps) = 0;

protected:
    int L{64};
    std::string prefix;
    unsigned int num_fields{1};

    MMSP::grid<dim, MMSP::vector<double> > *grid_ptr{nullptr};
};
#endif