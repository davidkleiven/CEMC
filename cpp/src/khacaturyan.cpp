#include "khachaturyan.hpp"
#include "use_numpy.hpp"
#include "additional_tools.hpp"
#include <omp.h>

using namespace std;
Khachaturyan::Khachaturyan(PyObject *ft_shape_func, PyObject *elastic_tensor, PyObject *misfit_strain){
    elastic.from_numpy(elastic_tensor);
    convertMisfit(misfit_strain);
    convertShapeFunc(ft_shape_func);
}

void Khachaturyan::convertMisfit(PyObject *pymisfit){
    PyObject *npy = PyArray_FROM_OTF(pymisfit, NPY_DOUBLE, NPY_IN_ARRAY);

    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        double *val = static_cast<double*>(PyArray_GETPTR2(npy, i, j));
        misfit[i][j] = *val;
    }
    Py_DECREF(npy);
}

void Khachaturyan::convertShapeFunc(PyObject *ft_shp){
    PyObject *npy = PyArray_FROM_OTF(ft_shp, NPY_DOUBLE, NPY_IN_ARRAY);

    npy_intp* dims = PyArray_DIMS(npy);

    for (unsigned int i=0;i<dims[0];i++)
    {
        vector< vector<double> > vec_outer;
        for (unsigned int j=0;j<dims[1];j++){
            vector<double> vec_inner;
            for (unsigned int k=0;k<dims[2];k++){
                double *val = static_cast<double*>(PyArray_GETPTR3(npy, i, j, k));
                vec_inner.push_back(*val);
            }
            vec_outer.push_back(vec_inner);
        }
        ft_shape_func.push_back(vec_outer);
    } 
    Py_DECREF(npy);
}

void Khachaturyan::green_function(mat3x3 &G, double direction[3]) const{
    mat3x3 Q;
    for (unsigned int i=0;i<3;i++)
    for (unsigned int p=0;p<3;p++)
    {
        Q[i][p] = 0.0;
        for (unsigned int j=0;j<3;j++)
        for (unsigned int l=0;l<3;l++){
            Q[i][p] += elastic(i, j, l, p)*direction[j]*direction[l];
        }
    }
    inverse3x3(Q, G);
}

PyObject* Khachaturyan::green_function(PyObject *direction) const{
    mat3x3 G;
    PyObject *npy_in = PyArray_FROM_OTF(direction, NPY_DOUBLE, NPY_IN_ARRAY);
    double *dir = static_cast<double*>(PyArray_GETPTR1(npy_in, 0));
    green_function(G, dir);
    Py_DECREF(npy_in);

    npy_intp dims[2] = {3, 3};
    PyObject *npy_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        double *val = static_cast<double*>(PyArray_GETPTR2(npy_out, i, j));
        *val = G[i][j];
    }
    return npy_out;
}

void Khachaturyan::effective_stress(mat3x3 &eff_stress) const{
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        eff_stress[i][j] = 0.0;
        for (unsigned int k=0;k<3;k++)
        for (unsigned int l=0;l<3;l++){
            eff_stress[i][j] += elastic(i, j, k, l)*misfit[k][l];
        }
    }
}

void Khachaturyan::wave_vector(unsigned int indx[3], double vec[3]) const{
    // Return the frequency follow Numpy conventions
    int sizes[3];
    sizes[0] = ft_shape_func.size();
    sizes[1] = ft_shape_func[0].size();
    sizes[2] = ft_shape_func[0][0].size();

    for(int i=0;i<3;i++){
        if (indx[i] < sizes[i]/2){
            vec[i] = static_cast<double>(indx[i])/sizes[i];
        }
        else{
            vec[i] = -1.0 + static_cast<double>(indx[i])/sizes[i];
        }
    }
}

double Khachaturyan::zeroth_order_integral(){
    mat3x3 eff_stress;
    effective_stress(eff_stress);
    
    double integral = 0.0;
    unsigned int nx = ft_shape_func.size();
    unsigned int ny = ft_shape_func[0].size();
    unsigned int nz = ft_shape_func[0][0].size();

    #ifdef PARALLEL_KHACHATURYAN_INTEGRAL
    #pragma omp parallel for collapse(3) reduction(+:integral)
    #endif
    for (unsigned int i=0;i<nx;i++)
    for (unsigned int j=0;j<ny;j++)
    for (unsigned int k=0;k<nz;k++)
    {
        // Handle this case separately!
        if ((i==0) && (j==0) && (k==0)){
            // G is not continuos
            // Average over 8 directions
            double weight = 1.0/8.0;
            double shape_val = ft_shape_func[0][0][0];
            for (int x=-1;x<=1;x+=2)
            for (int y=-1;y<=1;y+=2)
            for (int z=-1;z<=1;z+=2){
                unsigned int indx[3] = {1, 0, 0};
                double kvec[3];
                wave_vector(indx, kvec);
                mat3x3 G;
                green_function(G, kvec);
                double res = contract_green_function(G, eff_stress, kvec);
                integral += weight*res*shape_val;
            }
        }
        else{
            unsigned int indx[3] = {i, j, k};
            double kvec[3];
            wave_vector(indx, kvec);
            unit_vector(kvec);
            mat3x3 G;
            green_function(G, kvec);

            double res = contract_green_function(G, eff_stress, kvec);
            integral += res*ft_shape_func[i][j][k];
        }
    }
    return integral;
}

void Khachaturyan::unit_vector(double vec[3]){
    double length  = 0.0;
    for (unsigned int i=0;i<3;i++){
        length += vec[i]*vec[i];
    }

    length = sqrt(length);
    if (length < 1E-8) return;

    for (unsigned int i=0;i<3;i++){
        vec[i] /= length;
    }
}

double Khachaturyan::contract_green_function(const mat3x3 &G, const mat3x3 &eff_stress, double uvec[3]){
    double result = 0.0;
    double temp_vec[3] = {0.0, 0.0, 0.0};

    // Stress with unit vector
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        temp_vec[i] += eff_stress[i][j]*uvec[j];
    }

    // Green function with temp_vec
    double temp_vec2[3] = {0.0, 0.0, 0.0};
    for (unsigned int i=0;i<3;i++)
    for (unsigned int j=0;j<3;j++){
        temp_vec2[i] += G[i][j]*temp_vec[j];
    }

    for (unsigned int i=0;i<3;i++){
        result += temp_vec[i]*temp_vec2[i];
    }
    return result;
}