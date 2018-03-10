#ifndef LINEAR_VIB_CORRECTION_H
#define LINEAR_VIB_CORRECTION_H
#include <map>
#include <string>

class LinearVibCorrection
{
public:
  LinearVibCorrection( const std::map<std::string,double> &eci_per_kbT );

  /** Computes the contribution to the energy (per atom) from the vibrations */
  double energy( const std::map<std::string,double> &cf, double T ) const;
private:
  std::map<std::string,double> eci_per_kbT;
};
#endif
