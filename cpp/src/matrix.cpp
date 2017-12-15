#include "matrix.hpp"

Matrix::Matrix( unsigned int n_rows, unsigned int n_cols ):n_rows(n_rows), n_cols(n_cols)
{
  data = new double[n_rows*n_cols];
}

Matrix::~Matrix()
{
  delete [] data;
}

double& Matrix::operator()( unsigned int i, unsigned int j )
{
  return data[j*n_rows+i];
}

const double& Matrix::operator()( unsigned int i, unsigned int j ) const
{
  return data[j*n_rows+i];
}

void Matrix::set_size( unsigned int nr, unsigned int nc )
{
  n_rows = nr;
  n_cols = nc;
  data = new double[n_rows*n_cols];
}
