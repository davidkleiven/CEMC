#ifndef SYMBOLS_H
#define SYMBOLS_H
#include <vector>
#include <string>
#include <map>
#include <set>

typedef std::vector<std::string> vec_str_t;
typedef std::set<std::string> set_str_t;
typedef std::map<std::string, unsigned int> dict_uint_t;
class Symbols
{
public:
    Symbols(const vec_str_t &symbs, const set_str_t &unique_symbs);
    Symbols(const Symbols &other);
    Symbols& operator =(const Symbols &other);
    ~Symbols();

    /** Return the symbol ID of the atom at site indx */
    unsigned int id(unsigned int indx) const {return symb_ids[indx];};

    /** Check if the symb_id and symbols are consistent */
    bool is_consistent() const;

    /** Set a new symbol */
    void set_symbol(unsigned int indx, const std::string &symb);

    /** Get symbol at position n*/
    const std::string& get_symbol(unsigned int n) const {return symbols[n];};

    /** Get the array of symbols */
    const vec_str_t& get_symbols() const {return symbols;};

    /** Return the size of the symbol container */
    unsigned int size() const {return symbols.size();};

    /** Re-initialize all symbols */
    void set_symbols(const vec_str_t& new_symbs);

    /** Get the ID of a particular symbol */
    unsigned int get_symbol_id(const std::string &symb) const{return symb_id_translation.at(symb);};

    /** Return the number of uniquee symbols */
    unsigned int num_unique_symbols() const{return symb_id_translation.size();};
private:
    unsigned int *symb_ids{nullptr};
    vec_str_t symbols;
    dict_uint_t symb_id_translation;

    /** Syncronize the IDs with the symbols vector */
    void update_ids();

    /** Transfer members to other class */
    void swap(Symbols &other) const;
};
#endif