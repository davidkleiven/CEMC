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

    /** Set upper and lower bounds */
    void set_bounds(double lower, double upper);

    /** Set the penalty coefficient */
    void set_penalty(double new_penalty){penalty = new_penalty;};

    /** Scale for how fast the expoential penalty increase */
    void set_range_scale(double scale){rng_scale = scale;};

    /** Return true if the value is outside the specified range */
    bool is_outside_range(double x) const;

    /** Return the regularization term */
    double regularization(double x) const;

    /** Return the derivative of the regularization term */
    double regularization_deriv(double x) const;
private:
    std::vector<double> coeff;
    bool has_bounds{false};
    double lower_bound{0.0};
    double upper_bound{1.0};
    double penalty{100.0};
    double rng_scale{0.1};
};
#endif