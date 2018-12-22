#include "row_sparse_struct_matrix.hpp"
#include "additional_tools.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

void RowSparseStructMatrix::set_size( unsigned int n_rows, unsigned int n_non_zero_per_row, unsigned int max_lut_value )
{
  deallocate();
  lookup = new int[max_lut_value+1];
  max_lookup_value = max_lut_value;

  for ( unsigned int i=0;i<max_lookup_value+1;i++ )
  {
    lookup[i] = -1;
  }

  values = new int*[n_rows];
  for ( unsigned int i=0;i<n_rows;i++ )
  {
    values[i] = new int[n_non_zero_per_row];
  }
  num_non_zero = n_non_zero_per_row;
  allowed_lookup_values = new int[num_non_zero];
}

RowSparseStructMatrix::RowSparseStructMatrix(const RowSparseStructMatrix &other)
{
  this->swap(other);
}

RowSparseStructMatrix& RowSparseStructMatrix::operator=(const RowSparseStructMatrix &other){
  this->swap(other);
  return *this;
}

void RowSparseStructMatrix::deallocate()
{
  delete [] allowed_lookup_values;
  delete [] lookup;
  for (unsigned int i=0;i<num_rows;i++){
    delete [] values[i];
  }
  delete [] values;

  allowed_lookup_values = nullptr;
  lookup = nullptr;
  values = nullptr;
  num_rows = 0;
  max_lookup_value = 0;
  num_non_zero = 0;
  lut_values_set = false;
}

void RowSparseStructMatrix::set_size( unsigned int n_rows, unsigned int n_non_zero_per_row )
{
  set_size( n_rows, n_non_zero_per_row, n_rows );
}

void RowSparseStructMatrix::set_lookup_values( const vector<int> &lut_values )
{
  if ( lut_values_set )
  {
    throw logic_error( "Cannot modify the allowed lookup values. This has already been done, and they can't be modified!" );
  }
  lut_values_set = true;
  
  memcpy(allowed_lookup_values, &lut_values[0], lut_values.size()*sizeof(int));

  unsigned int max_value = *max_element(lut_values.begin(), lut_values.end() );
  if ( max_value > max_lookup_value )
  {
    throw invalid_argument( "The maximum lookup value exceeds the number given when the size was specified!" );
  }

  if ( lut_values.size() > num_non_zero )
  {
    throw invalid_argument( "The number of lookup values exceeds the number of entries stored!" );
  }

  for ( unsigned int i=0;i<lut_values.size();i++ )
  {
    lookup[allowed_lookup_values[i]] = i;
  }
}

bool RowSparseStructMatrix::is_allowed_lut(int col) const
{
  for ( unsigned int i=0;i<num_non_zero;i++ )
  {
    if ( allowed_lookup_values[i] == col )
    {
      return true;
    }
  }
  return false;
}

void RowSparseStructMatrix::insert( unsigned int row, unsigned int col, int value )
{
  if ( !is_allowed_lut(col) )
  {
    string msg;
    invalid_col_msg(col,msg);
    throw invalid_argument( msg );
  }

  values[row][lookup[col]] = value;
}

int& RowSparseStructMatrix::operator()( unsigned int row, unsigned int col )
{
  return values[row][lookup[col]];
}

const int& RowSparseStructMatrix::operator()( unsigned int row, unsigned int col ) const
{
  return values[row][lookup[col]];
}

int RowSparseStructMatrix::get_with_validity_check( unsigned int row, unsigned int col ) const
{
  if ( row >= num_rows )
  {
    stringstream ss;
    ss << "The row argument exceeds the maximum number of rows in the matrix!\n";
    ss << "Given: " << row << ". Maximum size: " << num_rows << endl;
    throw invalid_argument(  ss.str() );
  }

  if ( !is_allowed_lut(col) )
  {
    string msg;
    invalid_col_msg(col,msg);
    throw invalid_argument( msg );
  }
  return values[row][lookup[col]];
}

void RowSparseStructMatrix::invalid_col_msg( unsigned int col_provided, string &msg ) const
{
  stringstream ss;
  ss << "The column requested is not a valid column!\n";
  ss << "Given: " << col_provided << endl;
  ss << "Allowed lookup values:\n";
  for (unsigned int i=0;i<num_non_zero;i++)
  {
      ss << allowed_lookup_values[i] << endl;
  }
  msg = ss.str();
}

void RowSparseStructMatrix::swap(const RowSparseStructMatrix &other)
{
  this->num_rows = other.num_rows;
  this->max_lookup_value = other.max_lookup_value;
  this->num_non_zero = other.num_non_zero;

  this->allowed_lookup_values = new int[num_non_zero];
  this->lookup = new int[num_non_zero];
  this->values = new int*[num_rows];
  for (unsigned int i=0;i<num_rows;i++)
  {
    this->values[i] = new int[num_non_zero];
    memcpy(this->values[i], other.values[i], num_non_zero*sizeof(int));
  }

  // Copy content
  memcpy(this->allowed_lookup_values, other.allowed_lookup_values, num_non_zero*sizeof(int));
  memcpy(this->lookup, other.lookup, num_non_zero*sizeof(int));
}