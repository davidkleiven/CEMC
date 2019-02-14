#ifndef POLYNOMIAL_TERM_H
#define POLYNOMIAL_TERM_H
#include <vector>

typedef std::vector<unsigned int> uintvec_t;

class PolynomialTerm{
public:
    PolynomialTerm(const uintvec_t &inner_power, unsigned int outer_power);
    PolynomialTerm(unsigned int dim, unsigned int inner_power, unsigned int outer_power);
    ~PolynomialTerm();

    /** Evaluate the term  */
    double evaluate(double x[]) const;

    /** Evaluate the derivative of the term with respect to one coordinate */
    double deriv(double x[], unsigned int crd) const;

    /** Return the dimension of the polynomial term*/
    unsigned int get_dim() const{return dim;};
private:
    unsigned int dim;
    unsigned int outer_power{1};
    unsigned int *inner_power{nullptr};
    double *centers{nullptr};

    /** Evaluate the inner sum */
    double evaluate_inner(double x[]) const;
};
#endif