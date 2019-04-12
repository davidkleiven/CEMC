#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <vector>
#include <string>

class SparseMatrix{
public:
    SparseMatrix(){};

    void clear();
    void insert(unsigned int row, unsigned int col, double value);

    void dot(const std::vector<double> &vec, std::vector<double> &res) const;

    void save(const std::string &fname) const;
private:
    std::vector<double> values;
    std::vector<unsigned int> row;
    std::vector<unsigned int> col;
};
#endif