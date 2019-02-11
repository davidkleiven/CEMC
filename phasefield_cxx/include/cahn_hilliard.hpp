#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H
#include <vector>

class CahnHilliard{
public:
    CahnHilliard(const std::vector<double> &coeff);

    double evaluate(double x) const;
private:
    std::vector<double> coeff;
};
#endif