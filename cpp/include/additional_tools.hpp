#ifndef ADDITIONAL_TOOLS_H
#define ADDITIONAL_TOOLS_H
#include <vector>
#include <iostream>
#include <map>
#include <set>
#include <array>
#include "cf_history_tracker.hpp"
#include <Python.h>


//class SymbolChange;

template<class key,class value>
std::ostream& operator <<(std::ostream &out, const std::map<key,value> &map );

std::ostream& operator << (std::ostream &out, const SymbolChange &symb );

std::ostream& operator <<(std::ostream &out, const std::array<SymbolChange,2> &move );

template<class T>
std::ostream& operator <<( std::ostream &out, const std::vector<T> &vec );

template<class T>
std::vector<T>& cyclic_permute( std::vector<T> &vec );

template<class T, unsigned int N>
std::ostream& operator <<(std::ostream &out, const std::array<T, N> &array);

template<class T>
void keys( std::map<std::string,T> &, std::vector<std::string> &keys );

template<class T>
void set2vector( const std::set<T> &s, std::vector<T> &v );

int kronecker(int i, int j);

PyObject* string2py(const std::string &string);
std::string py2string(PyObject *str);

#include "additional_tools.tpp"
#endif
