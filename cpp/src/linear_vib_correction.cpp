#include "linear_vib_correction.hpp"

const double kB = 8.6173303E-5; // Boltzmann constant in eV/K

using namespace std;
LinearVibCorrection::LinearVibCorrection( const map<string,double> &eci_per_kbT ): eci_per_kbT(eci_per_kbT){};

double LinearVibCorrection::energy( const map<string,double> &cf, double T ) const
{
  double E = 0.0;
  for ( auto iter=eci_per_kbT.begin(); iter != eci_per_kbT.end(); ++iter )
  {
    E += cf.at(iter->first)*iter->second*kB*T;
  }
  return E;
}
