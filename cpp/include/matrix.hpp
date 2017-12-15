#ifndef MATRIX_H
#define MATRIX_H

class Matrix
{
public:
  Matrix( unsigned int n_rows, unsigned int n_cols );
  Matrix();
  ~Matrix();
  double& operator()( unsigned int i, unsigned int j );
  const double& operator()( unsigned int i, unsigned int j ) const;
  void set_size( unsigned int n_rows, unsigned int n_cols );
private:
  unsigned int n_rows;
  unsigned int n_cols;
  double *data{nullptr};
};
#endif
