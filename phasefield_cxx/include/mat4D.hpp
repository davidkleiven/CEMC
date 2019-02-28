#ifndef MAT4D_H
#define MAT4D_H
#include <Python.h>

class Mat4D{
public:
    Mat4D(unsigned int n1, unsigned int n2, \
          unsigned int n3, unsigned int n4);
    Mat4D(const Mat4D &other);
    Mat4D(){};
    ~Mat4D();

    Mat4D& operator=(const Mat4D &other);

    double operator()(unsigned int i, unsigned int j, \
                      unsigned int k, unsigned int l) const;

    double& operator()(unsigned int i, unsigned int j, \
                       unsigned int k, unsigned int l);

    /** Return the total size of the matrix */
    unsigned int size() const {return n1*n2*n3*n4;};

    /** Initalize the array from a Numpy array */
    void from_numpy(PyObject *array);

    /** Return a numpy representation of the array */
    PyObject* to_numpy() const;
private:
    unsigned int n1{0};
    unsigned int n2{0};
    unsigned int n3{0};
    unsigned int n4{0};
    double *data{nullptr};

    inline unsigned int get_index(unsigned int i, unsigned int j, 
                                  unsigned int k, unsigned int l) const;

    void allocate(unsigned int nn1, unsigned int nn2, unsigned int nn3, unsigned int nn4);
    void swap(const Mat4D &other);
};
#endif