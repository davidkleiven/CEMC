#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>

typedef std::vector< std::vector<std::string> > name_list;
typedef std::vector< std::vector< <std::vector<double> > > cluster_list;
typedef std::map<std::string,double> bf_list;
typedef std::map<std::string,double> cf;

class CEUpdater
{
public:
  CEUpdater();

  /** Public attributes */
  std::vector<std::string> symbols;
  name_list cluster_names;
  cluster_list cluster_indx;
  bf_list basis_functions;
  cf corr_functions;
  std::vector< cf > old_corrFuncs;
};
#endif
