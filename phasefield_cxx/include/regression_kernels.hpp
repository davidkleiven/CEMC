#ifndef REGRESSION_KERNEL_H
#define REGRESSION_KERNEL_H
#include <string>
#include <Python.h>

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

        /** Get the name of the kernel */
        const std::string& get_name() const{return name;};

        /** Check if the point is outside the kernels support */
        virtual bool is_outside_support(double x) const{return false;};

        /** Get a Python dictionary with the parameter */
        virtual PyObject *to_dict() const;

        /** Set parameters from a dictionary */
        virtual void from_dict(PyObject *dict_repr);
    protected:
        double upper_limit{0.0};
        double lower_limit{0.0};
        std::string name{"default"};
};

class QuadraticKernel: public RegressionKernel{
    public:
        QuadraticKernel(double width);

        /** Evaluate the kernel */
        virtual double evaluate(double x) const override final;

        /** Calculate the derivative of the kernel */
        virtual double deriv(double x) const override final;

        /** Check if the point is outside the support of the kernel */
        virtual bool is_outside_support(double x) const override final;

        /** Get a Python dictionary representation of the parameters */
        virtual PyObject *to_dict() const override final;
    private:
        double width{1.0};

        /** Calculate the amplitude of the kernel */
        double amplitude() const{return 0.75/width;};
};

class GaussianKernel: public RegressionKernel{
    public:
        GaussianKernel(double std_dev);

        /** Evaluate the kernel */
        virtual double evaluate(double x) const override final;

        /** Calculate the derivative of the kernel */
        virtual double deriv(double x) const override final;

        /** Check if the value is outside the support */
        virtual bool is_outside_support(double x) const override final;

        /** Get a Python dictionary representation of the parameters */
        virtual PyObject *to_dict() const override final;
    private:
        double std_dev{1.0};
};
#endif