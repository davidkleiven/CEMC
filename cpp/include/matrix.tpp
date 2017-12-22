
template<class T>
Matrix<T>::Matrix( unsigned int n_rows, unsigned int n_cols ):n_rows(n_rows), n_cols(n_cols)
{
  data = new T[n_rows*n_cols];
};

template<class T>
Matrix<T>::~Matrix()
{
  delete [] data;
};

template <class T>
T& Matrix<T>::operator()( unsigned int i, unsigned int j )
{
  return data[j*n_rows+i];
};

template<class T>
const T& Matrix<T>::operator()( unsigned int i, unsigned int j ) const
{
  return data[j*n_rows+i];
};

template<class T>
void Matrix<T>::set_size( unsigned int nr, unsigned int nc )
{
  n_rows = nr;
  n_cols = nc;
  data = new T[n_rows*n_cols];
};

template<class T>
void swap( Matrix<T> &first, const Matrix<T> &second )
{
  first.n_rows = second.n_rows;
  first.n_cols = second.n_cols;
  first.data = new T[second.n_rows*second.n_cols];
  for ( unsigned int i=0;i<first.n_rows*first.n_cols;i++ )
  {
    first.data[i] = second.data[i];
  }
}

template<class T>
Matrix<T>::Matrix( const Matrix<T> &other )
{
  swap(*this,other);
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &other )
{
  swap(*this,other);
  return *this;
}
