#ifndef BASIS_FUNCTION_H
#define BASIS_FUNCTION_H
#include "symbols_with_numbers.hpp"
#include <map>
#include <vector>

typedef std::map<std::string, double> dict_dbl_t;
typedef std::vector<dict_dbl_t> bf_raw_t;

class BasisFunction
{
public:
    BasisFunction(const bf_raw_t &raw_bfs, const Symbols &symb_with_num);
    ~BasisFunction();

    /** Return the basis function value for a given decoration number and symbol ID */
    double get(unsigned int dec_num, unsigned int symb_id) const;
private:
    const Symbols *symb_ptr{nullptr};
    bf_raw_t raw_bf_data;
    double *bfs{nullptr};
    unsigned int num_bfs{0};
    unsigned int num_bf_values{0};

    /** Return the corresponding index into the flattened array */
    unsigned int get_index(unsigned int dec_num, unsigned int symb_id) const;
};

#endif