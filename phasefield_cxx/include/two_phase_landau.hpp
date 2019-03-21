#ifndef TWO_PHASE_LANDAU_H
#define TWO_PHASE_LANDAU_H
#include <vector>
#include "kernel_regressor.hpp"
#include "polynomial.hpp"

class TwoPhaseLandau{
    public:
        TwoPhaseLandau();

        /** Set the kernel regressor */
        void set_kernel_regressor(const KernelRegressor &regr);

        /** Set the polymial */
        void set_polynomial(const Polynomial &poly);

        /** Evaluate the shape polynomial*/
        double evaluate(double conc, const std::vector<double> &shape) const;

        /** Evaluate the derivative */
        double partial_deriv_conc(double conc, const std::vector<double> &shape) const;

        /** Partial derivative with respect to the shape variable */
        double partial_deriv_shape(double conc, const std::vector<double> &shape, unsigned int direction) const;
    private:
        const KernelRegressor *regressor{nullptr};
        const Polynomial *polynomial{nullptr};
};
#endif