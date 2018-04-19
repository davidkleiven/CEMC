#ifndef NAMED_ARRAY_H
#define NAMED_ARRAY_H
#include <vector>
#include <map>
#include <string>

/** This is a class that mimic a map.
Random access is slower, here linear in the number of elements
It is most efficent when two maps having exactly the same keys are
iterated over.
*/
class NamedArray
{
public:
  NamedArray(){};

  /* Get one element */
  const double& operator[](unsigned int i) const { return data[i]; };
  double& operator[](unsigned int i){ return data[i]; };

  /** Write access */
  double& operator[]( const std::string& name );

  /** Read access */
  double at( const std::string& name ) const;


  /** Get the name of index i */
  const std::string& name( unsigned int i ) const { return names[i]; };

  /** Get a vector with all names */
  const std::vector<std::string>& get_names() const { return names; };

  /** Dot product between to named arrays */
  double dot( const NamedArray& other ) const;

  /** Checks if the names of the two named array objects are equal */
  bool names_are_equal( const NamedArray &other ) const;

  /** Initialize the class */
  void init( std::map<std::string,double> &values, const std::vector<std::string> &elements_names );
  void init( std::map<std::string,double> &values );

  /** Returns the size of the array */
  unsigned int size() const { return data.size(); };

  /** Update the entry corresponding to name */
  void update( const std::string& name, double value );

  /** Sets the naming array and initialize the data array to zeros */
  void set_order( const std::vector<std::string> &keys );

  /** Counts the number of elements having a specific key */
  unsigned int count( const std::string &name ) const;
private:
  std::vector<double> data;
  std::vector<std::string> names;
};

#endif
