#include "test_multidirectional_khachaturyan.hpp"
#include "multidirectional_khachaturyan.hpp"

const unsigned int L = 128;

PyObject *field2npy(MMSP::grid<2, MMSP::vector<fftw_complex> > &gr, unsigned int field){
    npy_intp dims[2] = {L, L};
    PyObject* npy_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    for (unsigned int node=0;node<MMSP::nodes(gr);node++){
        MMSP::vector<int> pos = gr.position(node);
        double* val = static_cast<double*>(PyArray_GETPTR2(npy_array, pos[0], pos[1]));
        *val = gr(node)[field].re;
    }
    return npy_array;
}

PyObject* test_functional_derivative(PyObject *elastic, PyObject *misfit){
    Khachaturyan khach(2, elastic, misfit);

    MultidirectionalKhachaturyan multi(0.8);
    multi.add_model(khach, 0);

    
    MMSP::grid<2, MMSP::vector<fftw_complex> > gr(0, L, 0, L);
    MMSP::grid<2, MMSP::vector<fftw_complex> > grid_out(gr);

    vector<int> shape_fields;
    shape_fields.push_back(0);

    multi.functional_derivative(gr, grid_out, shape_fields);
    return field2npy(grid_out, 0);
}