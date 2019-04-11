#ifndef TWO_PHASE_LANDAU_H
#define TWO_PHASE_LANDAU_H
#include <vector>
#include "kernel_regressor.hpp"
#include "polynomial.hpp"

class TwoPhaseLandau{
    public:
        TwoPhaseLandau();

        /** Set the kernel regressor */
        void set_kernel_regressor(const KernelRegressor &regr){regressor = &regr;};

        /** Set the polymial */
        void set_polynomial(const Polynomial &poly){polynomial = &poly;};

        /** Set discontinuity jump */
        void set_discontinuity(double conc, double jump);

        /** Get the dimension of the polynomial */
        unsigned int get_poly_dim() const;

        /** Evaluate the shape polynomial*/
        double evaluate(double conc, const std::vector<double> &shape) const;
        double evaluate(double x[]) const;

        /** Evaluate the derivative */
        double partial_deriv_conc(double conc, const std::vector<double> &shape) const;
        double partial_deriv_conc(double x[]) const;

        /** Partial derivative with respect to the shape variable */
        double partial_deriv_shape(double conc, const std::vector<double> &shape, unsigned int direction) const;
        double partial_deriv_shape(double x[], unsigned int direction) const;

        /** Return a pointer to the regressor */
        const KernelRegressor* get_regressor() const{return regressor;};
    private:
        const KernelRegressor *regressor{nullptr};
        const Polynomial *polynomial{nullptr};

        /** Return contribution from discontiuity jump */
        //double jump_contrib(double conc) const {return conc >= disc_conc ? -disc_jump : 0.0;};
        double jump_contrib(double conc) const;
        double jump_contrib_deriv(double conc) const;

        double disc_jump{0.0};
        double disc_conc{0.0};
        double step_func_width{0.1};
};
#endif