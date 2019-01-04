#include "mat4D.hpp"
#include "use_numpy.hpp"
#include <stdexcept>

using namespace std;

Mat4D::Mat4D(unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4): \
    n1(n1), n2(n2), n3(n3), n4(n4){
        data = new double[n1*n2*n3*n4];
    }

Mat4D::~Mat4D(){
    delete [] data;
}

unsigned int Mat4D::get_index(unsigned int i, unsigned int j,
                              unsigned int k, unsigned int l) const{
                                  return i*n2*n3*n4 + j*n3*n4 + k*n4 + l;
                              }

void Mat4D::allocate(unsigned int nn1, unsigned int nn2, unsigned int nn3, unsigned int nn4){
    delete [] data;

    // Update the dimensions
    n1 = nn1;
    n2 = nn2;
    n3 = nn3;
    n4 = nn4;

    // Allocate new array with the correct size
    data = new double[size()];
}

void Mat4D::from_numpy(PyObject *array){
    PyObject* npy = PyArray_FROM_OTF(array, NPY_DOUBLE, NPY_IN_ARRAY);
    int ndim = PyArray_NDIM(npy);

    if (ndim != 4){
        Py_DECREF(npy);
        throw invalid_argument("Numpy array have to a 4 dimensional array!");
    }

    npy_intp* dims = PyArray_DIMS(npy);
    allocate(dims[0], dims[1], dims[2], dims[3]);

    for (unsigned int i=0;i<dims[0];i++)
    for (unsigned int j=0;j<dims[1];j++)
    for (unsigned int k=0;k<dims[2];k++)
    for (unsigned int l=0;l<dims[3];l++){
        double* value = static_cast<double*>(PyArray_GETPTR4(npy, i, j, k, l));
        data[get_index(i, j, k, l)] = *value;
    }
    Py_DECREF(npy);
}

PyObject* Mat4D::to_numpy() const{
    npy_intp dims[4] = {n1, n2, n3, n4};
    PyObject* npy = PyArray_SimpleNew(4, dims, NPY_DOUBLE);
    for (unsigned int i=0;i<dims[0];i++)
    for (unsigned int j=0;j<dims[1];j++)
    for (unsigned int k=0;k<dims[2];k++)
    for (unsigned int l=0;l<dims[3];l++){
        double* value = static_cast<double*>(PyArray_GETPTR4(npy, i, j, k, l));
        *value = data[get_index(i, j, k, l)];
    }
    return npy;
}