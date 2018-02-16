#include "mc_observers.hpp"
#include "ce_updater.hpp"
#include <cmath>

using namespace std;
PairCorrelation::PairCorrelation( CEUpdater &up ): MCObserver(up)
{
  /*
  CFHistoryTracker& hist = updater->get_history();
  map<string,double> &current = hist.get_current();

  for ( auto iter=current.begin(); iter != current.end(); ++iter )
  {
    if ( iter->first.substr(0,3) == "c2_" )
    {
      cf_sum[iter->first] = iter->second;
      cf_sum_squared[iter->first] = pow( iter->second,2 );
    }
  }*/
}

void PairCorrelation::execute()
{
  /*
  map<string,double> &current = updater->get_history().get_current();
  for ( auto iter=cf_sum.begin(); iter != cf_sum.end(); ++iter )
  {
    iter->second += current[iter->first];
    cf_sum_squared[iter->first] += pow( current[iter->first],2 );
  }*/
}
