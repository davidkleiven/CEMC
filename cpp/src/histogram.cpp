#include "histogram.hpp"

Histogram::Histogram( unsigned int Nbins, double Emin, double Emax ):Nbins(Nbins),Emin(Emin),Emax(Emax)
{
  hist.resize(Nbins);
  logdos.resize(Nbins);

  for ( unsigned int i=0;i<Nbins;i++ )
  {
    hist[i] = 0;
    logdos[i] = 0.0;
    known_structures[i] = false;
  }
}

double Histogram::get_energy( unsigned int bin ) const
{
  return Emin + ((Emax-Emin)*bin)/Nbins;
}

unsigned int Histogram::get_bin( double energy ) const
{
  return ( energy-Emin )*Nbins/(Emax-Emin);
}

void Histogram::update( unsigned int bin, double mod_factor )
{
  hist[bin] += 1;
  logdos[bin] += mod_factor;
}

bool Histogram::is_flat( double criteria ) const
{
  unsigned int mean = 0;
  unsigned int minimum = 100000000;
  for ( unsigned int i=0;i<hist.size();i++ )
  {
    if ( !known_structures[i] ) continue;

    mean += hist[i];
    if ( hist[i] < minimum ) minimum = hist[i];
  }

  double mean_dbl = static_cast<double>(mean)/hist.size();
  return minimum > criteria*mean_dbl;
}
