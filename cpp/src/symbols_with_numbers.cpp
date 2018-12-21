#include "symbols_with_numbers.hpp"

using namespace std;

Symbols::~Symbols(){
    delete [] symb_ids;
    symb_ids = nullptr;
}

Symbols::Symbols(const vec_str_t &symbs, const vec_str_t &unique_symbs): symbols(symbs){
    symb_ids = new unsigned int[symbs.size()];
    unsigned int current_id = 0;
    for (const string &symb: unique_symbs)
    {
        symb_id_translation[symb] = current_id++;
    }

    // Populate the symb_id array
    for (unsigned int i=0;i<symbs.size();i++)
    {
        symb_ids[i] = symb_id_translation[symbs[i]];
    }
}

bool Symbols::is_consistent() const
{
    for (unsigned int i=0;i<symbols.size();i++)
    {
        if (symb_ids[i] != symb_id_translation.at(symbols[i]))
        {
            return false;
        }
    }
    return true;
}

void Symbols::set_symbol(unsigned int indx, const string &new_symb)
{
    symbols[indx] = new_symb;
    symb_ids[indx] = symb_id_translation[new_symb];
}