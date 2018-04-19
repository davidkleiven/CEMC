#ifndef LINEAR_VIB_CORRECTION_H
#define LINEAR_VIB_CORRECTION_H
#include <unordered_map>
#include <map>
#include <string>
#include "named_array.hpp"

class LinearVibCorrection
{
public:
  LinearVibCorrection( const std::map<std::string,double> &eci_per_kbT );

  /** Computes the contribution to the energy (per atom) from the vibrations */
  double energy( const NamedArray &cf, double T ) const;
private:
  std::map<std::string,double> eci_per_kbT;
};
#endif
