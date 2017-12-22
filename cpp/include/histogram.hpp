#ifndef HISTROGRAM_H
#define HISTOGRAM_H
#include <vector>

class Histogram
{
public:
  Histogram( unsigned int Nbins, double Emin, double Emax );

  /** Returns the bin corresponding to the given energy */
  unsigned int get_bin( double energy ) const;

  /** Returns the energy corresponding to one bin */
  double get_energy( unsigned int bin ) const;

  /** Updates the histogram and the logdos */
  void update( unsigned int bin, double mod_factor );

  /** Checks if the histogram is flat */
  bool is_flat( double criteria ) const;
private:
  unsigned int Nbins{1};
  double Emin{0.0};
  double Emax{1.0};

  std::vector<unsigned int> hist;
  std::vector<double> logdos;
  std::vector<bool> known_structures;
};
#endif
