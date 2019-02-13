#include "phase_field_simulation.hpp"
#include <stdexcept>
#include <sstream>
#include "MMSP.grid.h"
#include "MMSP.vector.h"
#include <ctime>

using namespace std;

template<int dim>
PhaseFieldSimulation<dim>::PhaseFieldSimulation(int L, \
                     const std::string &prefix, unsigned int num_fields): \
                     L(L), prefix(prefix), num_fields(num_fields){
                         if (dim == 1){
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields, 0, L);
                         }
                         else if (dim == 2){
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields, 0, L, 0, L);
                         }
                         else if (dim == 3){
                             grid_ptr = new MMSP::grid<dim, MMSP::vector<double> >(num_fields, 0, L, 0, L, 0, L);
                         }
                         srand(time(0));
                     };

template<int dim>
PhaseFieldSimulation<dim>::~PhaseFieldSimulation(){
    delete grid_ptr;
}

template<int dim>
void PhaseFieldSimulation<dim>::random_initialization(unsigned int field_no, double lower, double upper){

    double range = upper - lower;
    for (int i=0;i<MMSP::nodes(*this->grid_ptr);i++){
        (*grid_ptr)(i)[field_no] = range*rand()/static_cast<double>(RAND_MAX) + lower;
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

template<int dim>
void PhaseFieldSimulation<dim>::run(unsigned int start, unsigned int nsteps, int increment){
    for (unsigned int iter=start;iter<nsteps+start; iter+=increment){
        this->update(increment);

        // Generate output filename
        stringstream ss;
        ss << prefix << iter << ".vti";
        MMSP::output(*grid_ptr, ss.str().c_str());
    }
}

template<int dim>
void PhaseFieldSimulation<dim>::from_file(const std::string &fname){
    grid_ptr->input(fname.c_str(), 1, false);
}


// Explicit instatiations
template class PhaseFieldSimulation<1>;
template class PhaseFieldSimulation<2>;
template class PhaseFieldSimulation<3>;