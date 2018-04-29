#include "row_sparse_struct_matrix.hpp"
#include "additional_tools.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

void RowSparseStructMatrix::set_size( unsigned int n_rows, unsigned int n_non_zero_per_row, unsigned int max_lut_value )
{
  lookup.resize(max_lut_value+1);
  for ( unsigned int i=0;i<lookup.size();i++ )
  {
    lookup[i] = -1;
  }

  values.resize(n_rows);
  for ( unsigned int i=0;i<n_rows;i++ )
  {
    values[i].resize(n_non_zero_per_row);
  }
}

void RowSparseStructMatrix::set_size( unsigned int n_rows, unsigned int n_non_zero_per_row )
{
  set_size( n_rows, n_non_zero_per_row, n_rows );
}

void RowSparseStructMatrix::set_lookup_values( const vector<int> &lut_values )
{
  if ( allowed_lookup_values.size() > 0 )
  {
    throw logic_error( "Cannot modify the allowed lookup values. This has already been done, and they can't be modified!" );
  }
  allowed_lookup_values = lut_values;

  int max_value = *max_element(lut_values.begin(), lut_values.end() );
  if ( max_value > lookup.size() )
  {
    throw invalid_argument( "The maximum lookup value exceeds the number given when the size was specified!" );
  }

  if ( lut_values.size() > values[0].size() )
  {
    throw invalid_argument( "The number of lookup values exceeds the number of entries stored!" );
  }

  for ( unsigned int i=0;i<allowed_lookup_values.size();i++ )
  {
    lookup[allowed_lookup_values[i]] = i;
  }
}

bool RowSparseStructMatrix::is_allowed_lut( unsigned int col ) const
{
  for ( unsigned int i=0;i<allowed_lookup_values.size();i++ )
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
  if ( row >= values.size() )
  {
    stringstream ss;
    ss << "The row argument exceeds the maximum number of rows in the matrix!\n";
    ss << "Given: " << row << ". Maximum size: " << values.size() << endl;
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
  ss << allowed_lookup_values << endl;
  msg = ss.str();
}
