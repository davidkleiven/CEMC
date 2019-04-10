#include "sparse_matrix.hpp"

using namespace std;

void SparseMatrix::clear(){
    row.clear();
    col.clear();
    values.clear();
}

void SparseMatrix::insert(unsigned int r, unsigned int c, double value){
    row.push_back(r);
    col.push_back(c);
    values.push_back(value);
}


void SparseMatrix::dot(const vector<double> &vec, vector<double> &out) const{
    for (unsigned int i=0;i<values.size();i++){
        out[row[i]] += values[i]*vec[col[i]];
    }
}

