#ifndef MATRIX_H
#define MATRIX_H

template <class T>
class Matrix
{
public:
  Matrix( unsigned int n_rows, unsigned int n_cols );
  Matrix(){};
  ~Matrix();
  T& operator()( unsigned int i, unsigned int j );
  const T& operator()( unsigned int i, unsigned int j ) const;
  void set_size( unsigned int n_rows, unsigned int n_cols );
private:
  unsigned int n_rows;
  unsigned int n_cols;
  T *data{nullptr};
};

#include "matrix.tpp"
#endif
