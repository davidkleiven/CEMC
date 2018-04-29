#ifndef ADDITIONAL_TOOLS_H
#define ADDITIONAL_TOOLS_H
#include <vector>
#include <iostream>
#include <map>
#include <set>

class SymbolChange;

template<class key,class value>
std::ostream& operator <<(std::ostream &out, const std::map<key,value> &map );

std::ostream& operator << (std::ostream &out, const SymbolChange &symb );

template<class T>
std::ostream& operator <<( std::ostream &out, const std::vector<T> &vec );

template<class T>
std::vector<T>& cyclic_permute( std::vector<T> &vec );

template<class T>
void keys( std::map<std::string,T> &, std::vector<std::string> &keys );
#include "additional_tools.tpp"

template<class T>
void set2vector( const std::set<T> &s, std::vector<T> &v );
#endif
