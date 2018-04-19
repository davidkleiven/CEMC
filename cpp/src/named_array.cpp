#include "named_array.hpp"
#include "additional_tools.hpp"
#include <stdexcept>

using namespace std;

void NamedArray::init( map<string,double> &values, const vector<string> &element_names )
{
  data.resize(element_names.size());
  names.resize(data.size());
  for ( unsigned int i=0;i<element_names.size();i++ )
  {
    data[i] = values[element_names[i]];
    names[i] = element_names[i];
  }
}

void NamedArray::init( map<string,double> &values )
{
  vector<string> names;
  keys( values, names );
  init( values, names );
}

double NamedArray::dot( const NamedArray &other ) const
{
  double dot_prod = 0.0;
  for ( unsigned int i=0;i<data.size();i++ )
  {
    dot_prod += data[i]*other.data[i];
  }
  return dot_prod;
}

bool NamedArray::names_are_equal( const NamedArray &other ) const
{
  if ( other.data.size() != data.size() )
  {
    return false;
  }

  for ( unsigned int i=0;i<names.size();i++ )
  {
    if ( names[i] != other.names[i] )
    {
      return false;
    }
  }
  return true;
}

void NamedArray::update( const string& name, double value )
{
  for ( unsigned int i=0;i<names.size();i++ )
  {
    if ( name == names[i] )
    {
      data[i] = value;
    }
  }
  throw invalid_argument( "No name corresponding to "+name );
}

double& NamedArray::operator[]( const string& name )
{
  for ( unsigned int i=0;i<names.size();i++ )
  {
    if ( names[i] == name )
    {
      return data[i];
    }
  }
  throw invalid_argument( "No name corresponding to "+name );
}

double NamedArray::at( const string&name ) const
{
  for ( unsigned int i=0;i<names.size();i++ )
  {
    if ( name == names[i] )
    {
      return data[i];
    }
  }
  throw invalid_argument( "No name corresponding to "+name );
}

void NamedArray::set_order( const vector<string> &keys )
{
  names = keys;
  data.resize(names.size());
  for ( unsigned int i=0;i<data.size();i++ )
  {
    data[i] = 0.0;
  }
}

unsigned int NamedArray::count( const string&name ) const
{
  unsigned int counter = 0;
  for ( unsigned int i=0;i<names.size();i++ )
  {
    if ( names[i] == name )
    {
      counter += 1;
    }
  }
  return counter;
}
