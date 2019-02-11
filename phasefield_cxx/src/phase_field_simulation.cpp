#include "phase_field_simulation.hpp"
#include <stdexcept>
#include <sstream>
#include "MMSP.grid.h"
#include "MMSP.vector.h"

using namespace std;

template<int dim>
PhaseFieldSimulation<dim>::PhaseFieldSimulation(int L, \
                     const std::string &prefix, unsigned int num_fields): \
                     L(L), prefix(prefix), num_fields(num_fields){
                         if (dim == 1){
                             int min[1] = {0};
                             int max[1] = {L};
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields-1, min, max);
                         }
                         else if (dim == 2){
                             int min[2] = {0, 0};
                             int max[2] = {L, L};
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields-1, min, max);
                         }
                         else if (dim == 3){
                             int min[3] = {0, 0, 0};
                             int max[3] = {L, L, L};
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields-1, min, max);
                         }
                     };

template<int dim>
PhaseFieldSimulation<dim>::~PhaseFieldSimulation(){
    delete grid_ptr;
}

template<int dim>
void PhaseFieldSimulation<dim>::random_initialization(unsigned int field_no, double lower, double upper){

    double range = upper - lower;
    for (int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
        (*grid_ptr)(i)[field_no] = range*rand()/RAND_MAX + lower;
    }

    // Save grid
    stringstream ss;
    ss << prefix << ".grd";
    MMSP::output(*this->grid_ptr, ss.str().c_str());
}

template<int dim>
void PhaseFieldSimulation<dim>::random_initialization(double lower, double upper){
    random_initialization(0, lower, upper);
}


// Explicit instatiations
template class PhaseFieldSimulation<1>;
template class PhaseFieldSimulation<2>;
template class PhaseFieldSimulation<3>;