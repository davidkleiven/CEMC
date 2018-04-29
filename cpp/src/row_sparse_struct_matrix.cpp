#include "row_sparse_struct_matrix.hpp"
#include "additional_tools.hpp"
#include <stdexcept>
#include <sstream>

using namespace std;

void RowSparseStructMatrix::set_size( unsigned int n_rows, unsigned int n_non_zero_per_row, unsigned int max_lut_value )
{
  lookup.resize(max_lut_value);
  for ( unsigned int i=0;i<max_lut_value;i++ )
  {
    lookup[i] = -1;
  }

  values.resize(n_rows);
  for ( unsigned int i=0;i<n_rows;i++ )
  {
    values[i].resize(n_non_zero_per_row);
  }
}

void RowSparseStructMatrix::set_lookup_values( const vector<int> &lut_values )
{
  if ( allowed_lookup_values.size() > 0 )
  {
    throw logic_error( "Cannot modify the allowed lookup values. This has already been done, and they can't be modified!" );
  }
  allowed_lookup_values = lut_values;
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
    throw invalid_argument( "The provided column index is not among the allowed values!" );
  }

  values[row][lookup[col]] = value;
}

int& RowSparseStructMatrix::operator()( unsigned int row, unsigned int col )
{
  if ( !is_allowed_lut(col) )
  {
    throw invalid_argument( "The provided column index is not among the allowed values!" );
  }
  return values[row][lookup[col]];
}

int RowSparseStructMatrix::operator()( unsigned int row, unsigned int col ) const
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
    stringstream ss;
    ss << "The column requested is not a valid column!\n";
    ss << "Given: " << col << endl;
    ss << "Allowed lookup values:\n";
    ss << allowed_lookup_values << endl;
    throw invalid_argument( ss.str() );
  }
  return values[row][lookup[col]];
}
