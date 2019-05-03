#include "sparse_matrix.hpp"
#include <fstream>
#include <map>
#include <utility>
#include <cmath>
#include <algorithm>

using namespace std;

void SparseMatrix::clear(){
    converted_to_csr = false;
    num_rows = 0;
    row.clear();
    col.clear();
    values.clear();
}

void SparseMatrix::insert(unsigned int r, unsigned int c, double value){
    converted_to_csr = false;
    row.push_back(r);
    col.push_back(c);
    values.push_back(value);

    if (r + 1 < num_rows){
        num_rows = r + 1;
    }
}


void SparseMatrix::dot(const vector<double> &vec, vector<double> &out) const{

    // TODO: Parallilize this loop. Remember that two processes
    // cannot update the same index of out simultaneously
    // #pragma omp declare reduction(elem_wise_sum: vector<double> : \
    //                               transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plus<double>())) \
    //                               initializer(omp_priv=omp_orig)
    
    // #ifndef NO_PHASEFIELD_PARALLEL
    // #pragma omp parallel for reduction(elem_wise_sum : out)
    // #endif
    // for (unsigned int i=0;i<values.size();i++){
    //     out[row[i]] += values[i]*vec[col[i]];
    // }

    if (!converted_to_csr){
        throw invalid_argument("The sparse matrix has not been converted to CSR!");
    }

    #ifndef NO_PHASEFIELD_PARALLEL
    #pragma omp parallel for
    #endif
    for (unsigned int r=0;r<num_rows;r++){
        out[r] = 0.0;
        for (unsigned int j=row_ptr[r];j<row_ptr[r+1];j++){
            out[r] += csr_values[j]*vec[csr_col_indx[j]];
        }
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

void SparseMatrix::count_entries_on_each_row(std::vector<unsigned int> &num_entries) const{
    for (auto row : row){
        num_entries[row] += 1;
    }
}

unsigned int SparseMatrix::max_row() const{
    unsigned int max = 0;
    for (auto r : row){
        if (r > max){
            max = r;
        }
    }
    return max;
}

void SparseMatrix::to_csr(){
    unsigned int max_r = max_row();
    vector<unsigned int> num_each_row(max_r+1);
    num_rows = max_r + 1;

    fill(num_each_row.begin(), num_each_row.end(), 0);
    count_entries_on_each_row(num_each_row);

    // Set up the row pointer
    row_ptr.resize(max_r+2);
    row_ptr[0] = 0;
    for (unsigned int i=0;i<num_each_row.size();i++){
        row_ptr[i+1] = row_ptr[i] + num_each_row[i];
    }

    csr_col_indx.resize(values.size());
    csr_values.resize(values.size());
    fill(csr_col_indx.begin(), csr_col_indx.end(), -1);

    // Fill column index
    for (unsigned int i=0;i<values.size();i++){
        unsigned int r_start = row_ptr[row[i]];
        unsigned int r_end = row_ptr[row[i]+1];
        bool found = false;

        for (unsigned int j=r_start;j<r_end;j++){
            if (csr_col_indx[j] == -1){
                csr_col_indx[j] = col[i];
                csr_values[j] = values[i];
                found = true;
                break;
            }
        }

        if (!found){
            throw runtime_error("Could not convert values to CSR format!");
        }
    }
    converted_to_csr = true;
}