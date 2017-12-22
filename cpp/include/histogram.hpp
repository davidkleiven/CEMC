#ifndef HISTROGRAM_H
#define HISTOGRAM_H
#include <vector>
#include <Python.h>

class Histogram
{
public:
  Histogram( unsigned int Nbins, double Emin, double Emax );

  /** Returns the bin corresponding to the given energy */
  int get_bin( double energy ) const;

  /** Returns the energy corresponding to one bin */
  double get_energy( int bin ) const;

  /** Updates the histogram and the logdos */
  void update( unsigned int bin, double mod_factor );

  /** Checks if the histogram is flat */
  bool is_flat( double criteria ) const;

  /** Returns the ratio of the DOS at the old_bin and at new_bin */
  double get_dos_ratio_old_divided_by_new( unsigned int old_bin, unsigned int new_bin ) const;

  /** Returns true if the bin is in the histogram range */
  bool bin_in_range( int bin ) const;

  /** Sends the result to the Python histograms */
  void send_to_python_hist( PyObject *py_hist );
private:
  unsigned int Nbins{1};
  double Emin{0.0};
  double Emax{1.0};

  std::vector<unsigned int> hist;
  std::vector<double> logdos;
  std::vector<bool> known_structures;
};
#endif
