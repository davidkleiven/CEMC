#include "polynomial.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

void Polynomial::add_term(double new_coeff, const PolynomialTerm &new_term){
    if (new_term.get_dim() != dim){
        stringstream ss;
        ss << "Dimension of new term does not match the dimension ";
        ss << "of the polynomial. Polynomial dimension: " << dim;
        ss << " dimension of new term " << new_term.get_dim();
        throw invalid_argument(ss.str());
    }

    terms.push_back(new_term);
    coeff.push_back(new_coeff);
}

double Polynomial::evaluate(double x[]) const{
    double value = 0.0;
    for (unsigned int i=0;i<terms.size();i++){
        value += coeff[i]*terms[i].evaluate(x);
    }
    return value;
}

double Polynomial::deriv(double x[], unsigned int crd) const{
    double value = 0.0;
    for (unsigned int i=0;i<terms.size();i++){
        value += coeff[i]*terms[i].deriv(x, crd);
    }
    return value;
}

ostream& operator <<(ostream& out, const Polynomial &instance){
    for (unsigned int i=0;i<instance.coeff.size();i++){
        out << "Coeff: " << instance.coeff[i] << " " << instance.terms[i] << "\n";
    }
    return out;
}