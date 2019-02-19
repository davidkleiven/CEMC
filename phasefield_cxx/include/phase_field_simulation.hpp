#ifndef PHASE_FIELD_SIMULATION_H
#define PHASE_FIELD_SIMULATION_H
//#include "MMSP.grid.h"
//#include "MMSP.vector.h"
#include "MMSP.grid.h"
#include "MMSP.vector.h"

#include <string>
#include <vector>
#include <Python.h>

template<int dim>
class PhaseFieldSimulation{
public:
    PhaseFieldSimulation(int L, \
                         const std::string &prefix, unsigned int num_fields);
    virtual ~PhaseFieldSimulation();

    /** Initialize a given field with random numbers */
    void random_initialization(unsigned int field_no, double lower, double upper);
    void random_initialization(double lower, double upper);

    /** Load grid from file */
    void from_file(const std::string &fname);

    /** Initialize the simulation from a list of Numpy arrays */
    void from_npy_array(PyObject *npy_arrays);

    /** Convert fields to list of Numpy Arrays */
    PyObject* to_npy_array() const;

    /** Update function */
    virtual void update(int steps) = 0;

    /** Run the simulation */
    void run(unsigned int start, unsigned int nsteps, int increment);
protected:
    int L{64};
    std::string prefix;
    unsigned int num_fields{1};
    unsigned int num_digits_in_file{10};

    /** Get iteration identifier with 10 digits */
    std::string get_digit_string(unsigned int iter) const;

    /** Initialize the given field from a Numpy array */
    void init_field_from_npy_arr(unsigned int field, PyObject *npy_arr);

    /** Convert a field to Numpy array */
    PyObject* field_to_npy_arr(unsigned int field) const;

    MMSP::grid<dim, MMSP::vector<double> > *grid_ptr{nullptr};
};
#endif