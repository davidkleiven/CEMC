#ifndef REGRESSION_KERNEL_H
#define REGRESSION_KERNEL_H

class RegressionKernel{
    public:
        RegressionKernel(){};

        /** Evaluate the kernel */
        virtual double evaluate(double x) const = 0;
        
        /** Calculate the derivative of the kernel */
        virtual double deriv(double x) const = 0;

        /** Upper limit of the support */
        double upper() const{return upper_limit;};

        /** Lower limit */
        double lower() const{return lower_limit;};
    protected:
        double upper_limit{0.0};
        double lower_limit{0.0};
};

class QuadraticKernel: public RegressionKernel{
    public:
        QuadraticKernel(double width);

        /** Evaluate the kernel */
        virtual double evaluate(double x) const override final;

        /** Calculate the derivative of the kernel */
        virtual double deriv(double x) const override final;

        /** Check if the point is outside the support of the kernel */
        bool is_outside_support(double x) const;
    private:
        double width{1.0};

        /** Calculate the amplitude of the kernel */
        double amplitude() const{return 0.75/width;};
};
#endif