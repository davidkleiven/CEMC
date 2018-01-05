#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <vector>
#include <Python.h>
#define DEBUG_LOSS_OF_PRECISION

class Histogram
{
public:
  Histogram( unsigned int Nbins, double Emin, double Emax );
  virtual ~Histogram(){};

  /** Returns the bin corresponding to the given energy */
  int get_bin( double energy ) const;

  /** Returns the energy corresponding to one bin */
  double get_energy( int bin ) const;

  /** Return the minimum energy of the histogram */
  double get_emin() const { return Emin; }

  /** Returns the maximum energy of the histogram */
  double get_emax() const { return Emax; };

  /** Returns the number of bins */
  unsigned int get_nbins() const { return Nbins; };

  /** Updates the histogram and the logdos */
  virtual void update( unsigned int bin, double mod_factor );

  /** Checks if the histogram is flat */
  virtual bool is_flat( double criteria );

  /** Returns the ratio of the DOS at the old_bin and at new_bin */
  double get_dos_ratio_old_divided_by_new( unsigned int old_bin, unsigned int new_bin ) const;

  /** Returns true if the bin is in the histogram range */
  virtual bool bin_in_range( int bin ) const;

  /** Sends the result to the Python histograms */
  void send_to_python_hist( PyObject *py_hist );

  /** Resets the histogram */
  virtual void reset();

  /** Read histogram data from Python histogram */
  void init_from_pyhist( PyObject *pyhist );

  const std::vector<unsigned int>& get_histogram() const { return hist; };
  const std::vector<double>& get_logdos() const { return logdos; };
protected:
  unsigned int Nbins{1};
  double Emin{0.0};
  double Emax{1.0};
  bool track_states{false};

  std::vector<unsigned int> hist;
  std::vector<double> logdos;
  std::vector<bool> known_structures;
};
#endif
