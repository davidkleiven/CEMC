#ifndef TWO_PHASE_LANDAU_H
#define TWO_PHASE_LANDAU_H
#include <vector>

class TwoPhaseLandau{
    public:
        TwoPhaseLandau(double c1, double c2, 
                                 const std::vector<double> &coeff);

        /** Evaluate the shape polynomial*/
        double evaluate(double conc, const std::vector<double> &shape) const;
    private:
        double c1{0.0};
        double c2{0.0};
        std::vector<double> coeff;
};
#endif