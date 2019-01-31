#ifndef KHACHATURYAN_H
#define KHACHATURYAN_H
#include <Python.h>
#include <array>
#include <vector>
#include "mat4D.hpp"

typedef std::array< std::array<double, 3>, 3> mat3x3;
typedef std::vector< std::vector< std::vector<double> > > shp_t;

class Khachaturyan{
public:
    Khachaturyan(PyObject *ft_shape_func, PyObject *elastic_tensor, PyObject *misfit_strain);

    /** Calculate the green function (omitting normalization factor 1/k^2)*/
    void green_function(mat3x3 &G, double direction[3]) const;
    PyObject* green_function(PyObject *direction) const;

    /** Effective stress tensor */
    void effective_stress(mat3x3 &eff_stress) const;

    /** Calculate the frequency corresponding to a set indx ({ix, iy, iz })*/
    void wave_vector(unsigned int indx[3], double vec[3]) const;

    /** Calculate the zeroth order integral */
    double zeroth_order_integral();
private:
    mat3x3 misfit;
    Mat4D elastic;
    shp_t ft_shape_func;

    void convertMisfit(PyObject *pymisfit);
    void convertShapeFunc(PyObject *ft_shp);
    static void unit_vector(double vec[3]);
    static double contract_green_function(const mat3x3 &G, const mat3x3 &eff_stress, double uvec[3]);
};
#endif