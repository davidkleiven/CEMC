#ifndef ROW_SPARSE_STRUCT_MATRIX_H
#define ROW_SPARSE_STRUCT_MATRIX_H
#include <vector>

/**
A Row Sparse Structured Matrix (RSSM) is a matrix that has non-zero entries in all
rows. So if the dense version is a NxM matrix this datastructures also has
N rows.
Moreover, in a RSSM the non-zero elements is located on the same positions
on each row.

Example of RSSM:
[0 1 0 0 10 0 20]
[0 2 0 0 11 0 32]
[0 7 0 0 43 0 51]
[0 2 0 0 21 0 13]

This matrix has 4 rows with 3 non-zero elements in each row located in the
same column.

The memory consumpiton of this data structure scales as
N x <number of non-zero entries in each row>
*/
class RowSparseStructMatrix
{
public:
  RowSparseStructMatrix(){};

  /** Initialize the size arrays */
  void set_size( unsigned int n_rows, unsigned int n_non_zero_per_row, unsigned int max_lut_value );

  /** Set the allowed lookup values */
  void set_lookup_values( const std::vector<int> &lut_values );

  /** Insert a new value */
  void insert( unsigned int row, unsigned int col, int value );

  /** Check if the provided value is allowed */
  bool is_allowed_lut( unsigned int col ) const;

  /** Matrix-like access operator. NOTE: For max performance this function does not perform any validity checks */
  int operator()( unsigned int row, unsigned int col ) const;

  /** Matrix-like write operator */
  int& operator()( unsigned int row, unsigned int col );

  /** Access function that verifies that the lookup is valid */
  int get_with_validity_check( unsigned int row, unsigned int col ) const;
private:
  std::vector<int> allowed_lookup_values;
  std::vector<int> lookup;
  std::vector< std::vector<int> > values;
};
#endif
