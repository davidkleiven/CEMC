#include "sparse_matrix.hpp"
#include <fstream>
#include <map>
#include <utility>
#include <cmath>
#include <algorithm>

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

    // TODO: Parallilize this loop. Remember that two processes
    // cannot update the same index of out simultaneously
    #pragma omp declare reduction(elem_wise_sum: vector<double> : \
                                  transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<double>())) \
                                  initializer(omp_priv=omp_orig)
    
    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for reduction(elem_wise_sum : out)
    #endif
    for (unsigned int i=0;i<values.size();i++){
        out[row[i]] += values[i]*vec[col[i]];
    }
}

void SparseMatrix::save(const string &fname) const{
    ofstream out(fname);

    if(!out.good()){
        return;
    }

    out << "# row, col, value\n";
    for (unsigned int i=0;i<values.size();i++){
        out << row[i] << ", " << col[i] << ", " << values[i] << "\n";
    }
    out.close();
}

bool SparseMatrix::is_symmetric() const{
    map<pair<unsigned int, unsigned int>, double> sorted;
    const double tol = 1E-6;
    for (unsigned int i=0;i<values.size();i++){
        pair<unsigned int, unsigned int> rowcol;
        unsigned int min_rowcol = row[i] < col[i] ? row[i] : col[i];
        unsigned int max_rowcol = row[i] >= col[i] ? row[i] : col[i];

        rowcol.first = min_rowcol;
        rowcol.second = max_rowcol;

        const auto iterator = sorted.find(rowcol);
        if (iterator != sorted.end()){
            if (abs(iterator->second - values[i]) > tol){
                return false;
            }
        }
        else{
            sorted[rowcol] = values[i];
        }
    }
    return true;
}

