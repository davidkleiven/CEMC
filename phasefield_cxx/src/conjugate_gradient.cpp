#include "conjugate_gradient.hpp"
#include "tools.hpp"
#include <iostream>

using namespace std;
ConjugateGradient::ConjugateGradient(double tol): tol(tol){};

void ConjugateGradient::solve(const SparseMatrix &mat, const vector<double> &rhs, vector<double> &x) const{
    vector<double> residual = rhs;
    vector<double> dotProd(x.size());
    fill(dotProd.begin(), dotProd.end(), 0.0);

    mat.dot(x, dotProd);
    inplace_minus(residual, dotProd);

    vector<double> p = residual;

    while (inf_norm(residual) > tol){
        fill(dotProd.begin(), dotProd.end(), 0.0);
        mat.dot(p, dotProd);

        double r_dot_r = dot(residual, residual);
        double alpha = r_dot_r/dot(p, dotProd);

        // Update x
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<x.size();i++){
            x[i] += alpha*p[i];
            residual[i] -= alpha*dotProd[i];
        }

        if (inf_norm(residual) < tol){
            break;
        }

        double beta = dot(residual, residual)/r_dot_r;

        // Update p
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<p.size();i++){
            p[i] = residual[i] + beta*p[i];
        }
    }
}