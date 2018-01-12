#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <vector>
#include <string>
#include <Python.h>
#include <array>
#define DEBUG_LOSS_OF_PRECISION

class Histogram
{
public:
  Histogram( unsigned int Nbins, double Emin, double Emax );
  Histogram( const Histogram &other );
  virtual ~Histogram();

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

  /** Returns the bin that is currently most visited */
  int get_current_max_bin() const { return current_max_bin; };

  /** Returns the bin that is currently least visited */
  int get_current_min_bin() const { return current_min_bin; };

  /** Returns the number of bins that are currently active */
  virtual unsigned int get_number_of_active_bins() const { return get_nbins(); };

  /** Updates the histogram and the logdos */
  virtual void update( unsigned int bin, double mod_factor );

  /** Updates the sub bin histograms */
  void update_sub_bin( unsigned int bin, double energy );

  /** Checks if the histogram is flat */
  virtual bool is_flat( double criteria );

  /** Returns the ratio of the DOS at the old_bin and at new_bin */
  double get_dos_ratio_old_divided_by_new( unsigned int old_bin, unsigned int new_bin ) const;

  /** Returns true if the bin is in the histogram range */
  virtual bool bin_in_range( int bin ) const;

  /** Allows histogram types supporting windowing to change the overlap */
  virtual void set_overlap( unsigned int overlap ){};

  /** Sends the result to the Python histograms */
  void send_to_python_hist( PyObject *py_hist );

  /** Resets the histogram */
  virtual void reset();

  /** Read histogram data from Python histogram */
  void init_from_pyhist( PyObject *pyhist );

  /** Stores the sub-bin distribution into a text file */
  void save_sub_bin_distribution( const std::string &fname ) const;

  /** Initialize the sub bin histograms */
  void init_sub_bins();

  /** Clears the sub bin histograms */
  void clear_sub_hist();

  /** Updates the from bin transfer */
  void update_bin_transfer( int from, int to );

  /** Stores bin transfer array to file */
  void save_bin_transfer( const std::string &fname ) const;

  /** Redistributes the random walkers. See Adaptive Window Histogram */
  virtual void redistribute_samplers(){};

  /** Returns true if the probability of data conflict is too large */
  virtual bool update_synchronized( unsigned int num_threads, double conflict_prob ) const;

  const std::vector<unsigned int>& get_histogram() const { return hist; };
  const std::vector<double>& get_logdos() const { return logdos; };
  friend void swap( Histogram &first, const Histogram &other );
protected:
  unsigned int Nbins{1};
  double Emin{0.0};
  double Emax{1.0};
  bool track_states{false};
  int current_max_bin{0};
  int current_min_bin{0};

  std::vector<unsigned int> hist;
  std::vector<double> logdos;
  std::vector<bool> known_structures;
  std::vector<Histogram*> sub_bin_distribution;
  std::vector< std::array<unsigned int,21> > bin_transfer;
  double avg_acc_rate{0.0};
};
#endif
