#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H
#include "polynomial_term.hpp"
#include <iostream>

class Polynomial{
public:
    Polynomial(unsigned int dim): dim(dim){};

    /** Add a new term to the polynomial*/
    void add_term(double coeff, const PolynomialTerm &new_term);

    /** Evaluate the polynomial */
    double evaluate(double x[]) const;

    /** Evaluate the derivative of the polynomial */
    double deriv(double x[], unsigned int crd) const;

    /** Return the dimension (number of variables) */
    unsigned int get_dim() const{return dim;};

    friend std::ostream& operator << (std::ostream& out, const Polynomial &instance);
private:
    unsigned int dim;
    std::vector<PolynomialTerm> terms;
    std::vector<double> coeff;
};

#endif