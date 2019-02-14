#ifndef POLYNOMIAL_TERM_H
#define POLYNOMIAL_TERM_H
#include <vector>

typedef std::vector<double> vec_t;

class PolynomialTerm{
public:
    PolynomialTerm(unsigned int dim, const vec_t &inner_power, unsigned int outer_power);
    PolynomialTerm(unsigned int dim, unsigned int inner_power, unsigned int outer_power);
    ~PolynomialTerm();

    /** Evaluate the term  */
    double evaluate(double x[]) const;

    /** Evaluate the derivative of the term with respect to one coordinate */
    double deriv(double x[], double crd) const;
private:
    unsigned int dim;
    unsigned int outer_power{1};
    unsigned int *inner_powers;
    double *centers;
    double evaluate_inner(double x[]) const;
};
#endif