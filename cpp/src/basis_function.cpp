#include "basis_function.hpp"
#include "additional_tools.hpp"

using namespace std;
BasisFunction::BasisFunction(const bf_raw_t &raw_bfs, const Symbols &symb_with_num): \
    raw_bf_data(raw_bfs), symb_ptr(&symb_with_num)
    {
        num_bfs = raw_bf_data.size();
        num_bf_values = symb_ptr->num_unique_symbols();
        bfs = new double[num_bfs*num_bf_values];

        // Transfer the raw bf array to the flattened array
        for (unsigned int dec_num=0;dec_num<num_bfs;dec_num++)
        for (auto iter=raw_bf_data[dec_num].begin(); iter != raw_bf_data[dec_num].end(); ++iter)
        {
            unsigned int indx = get_index(dec_num, symb_ptr->get_symbol_id(iter->first));
            bfs[indx] = iter->second;
        }
    };

BasisFunction::BasisFunction(const BasisFunction &other){
    this->swap(other);
}

BasisFunction& BasisFunction::operator=(const BasisFunction &other){
    this->swap(other);
    return *this;
}

BasisFunction::~BasisFunction(){
    delete [] bfs;
}

unsigned int BasisFunction::get_index(unsigned int dec_num, unsigned int symb_id) const
{
    return dec_num*num_bf_values + symb_id;
}

double BasisFunction::get(unsigned int dec_num, unsigned int symb_id) const
{
    return bfs[get_index(dec_num, symb_id)];
}

void BasisFunction::swap(const BasisFunction &other)
{
    this->raw_bf_data = other.raw_bf_data;
    this->symb_ptr = other.symb_ptr;
    this->num_bfs = other.num_bfs;
    this->num_bf_values = other.num_bf_values;
    if (this->bfs != nullptr) delete [] this->bfs;
    this->bfs = new double[num_bfs*num_bf_values];
    memcpy(this->bfs, other.bfs, num_bfs*num_bf_values*sizeof(double));
}

ostream& operator<<(ostream &out, const BasisFunction &bf)
{
    out << "Basis Function object\n";
    out << "Raw data\n";
    out << bf.raw_bf_data << "\n";
    out << "Flattened array\n";
    for (unsigned int i=0;i<bf.num_bfs*bf.num_bf_values;i++)
    {
        out << bf.bfs[i] << " ";
    }
    return out;
}