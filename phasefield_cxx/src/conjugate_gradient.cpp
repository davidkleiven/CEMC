#include "conjugate_gradient.hpp"
#include "tools.hpp"
#include <iostream>

using namespace std;
ConjugateGradient::ConjugateGradient(double tol): tol(tol){};

void ConjugateGradient::solve(const SparseMatrix &mat, const vector<double> &rhs, vector<double> &x) const{
    vector<double> residual = rhs;
    vector<double> Ax(x.size());
    vector<double> Ap(x.size());
    fill(Ax.begin(), Ax.end(), 0.0);

    mat.dot(x, Ax);
    inplace_minus(residual, Ax);

    if (inf_norm(residual) < tol){
        return;
    }

    vector<double> p = residual;

    double r_dot_r = dot(residual, residual);
    bool did_converge = false;
    for(unsigned int iter=0;iter<rhs.size();iter++){
        fill(Ap.begin(), Ap.end(), 0.0);
        mat.dot(p, Ap);

        //double r_dot_r = dot(residual, residual);
        double alpha = r_dot_r/dot(p, Ap);

        // Update x
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<x.size();i++){
            x[i] += alpha*p[i];
            residual[i] -= alpha*Ap[i];
        }

        if (inf_norm(residual) < tol){
            did_converge = true;
            break;
        }

        //double beta = dot(residual, residual)/r_dot_r;
        double r_dot_r_new = dot(residual, residual);

        // Update p
        #ifndef NO_PHASEFIELD_PARALLEL
        #pragma omp parallel for
        #endif
        for (unsigned int i=0;i<p.size();i++){
            //p[i] = residual[i] + beta*p[i];
            p[i] = residual[i] + r_dot_r_new*p[i]/r_dot_r;
        }
        r_dot_r = r_dot_r_new;
    }

    if (!did_converge){
        throw runtime_error("CG did not converge. This is a bug!");
    }
}