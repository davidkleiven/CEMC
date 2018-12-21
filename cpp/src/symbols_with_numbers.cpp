#include "symbols_with_numbers.hpp"
#include <cstring>

using namespace std;

Symbols::~Symbols(){
    delete [] symb_ids;
    symb_ids = nullptr;
}

Symbols::Symbols(const vec_str_t &symbs, const set_str_t &unique_symbs): symbols(symbs){
    symb_ids = new unsigned int[symbs.size()];
    unsigned int current_id = 0;
    for (auto iter=unique_symbs.begin(); iter != unique_symbs.end(); ++iter)
    {
        symb_id_translation[*iter] = current_id++;
    }

    // Populate the symb_id array
    update_ids();
}

Symbols::Symbols(const Symbols &other)
{
    other.swap(*this);
}

Symbols& Symbols::operator=(const Symbols &other){
    other.swap(*this);
    return *this;
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

void Symbols::set_symbols(const vec_str_t &new_symbs){
    delete [] symb_ids;
    symb_ids = new unsigned int[new_symbs.size()];
    symbols = new_symbs;
    update_ids();
}

void Symbols::update_ids()
{
    for (unsigned int i=0;i<symbols.size();i++)
    {
        symb_ids[i] = symb_id_translation[symbols[i]];
    }
}

void Symbols::swap(Symbols &other) const{
    other.symbols = symbols;
    other.symb_id_translation = symb_id_translation;

    delete [] other.symb_ids;
    other.symb_ids = new unsigned int[symbols.size()];
    memcpy(other.symb_ids, symb_ids, symbols.size()*sizeof(unsigned int));
}