#ifndef MC_OBSERVERS_H
#define MC_OBSERVERS_H
#include <map>
#include <string>

class CEUpdater; // Forward declaration

class MCObserver
{
public:
  MCObserver( CEUpdater &updater ): updater(&updater){};
  virtual ~MCObserver(){};

  /** Performs the action of the observer */
  virtual void execute() = 0;
protected:
  CEUpdater *updater;
};

class PairCorrelation: public MCObserver
{
public:
  PairCorrelation( CEUpdater &updater );

  /** Tracks the evolution of the correlation functions */
  void execute();
private:
  unsigned int n{1};
  std::map< std::string, double > cf_sum;
  std::map< std::string, double > cf_sum_squared;
};
#endif
