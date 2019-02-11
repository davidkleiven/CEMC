#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H
#include <vector>

class CahnHilliard{
public:
    CahnHilliard(const std::vector<double> &coeff);

    /** Evaluate the polynomial */
    double evaluate(double x) const;

    /** Evaluate the first derivative of the polynomial */
    double deriv(double x) const;
private:
    std::vector<double> coeff;
};
#endif