#ifndef CE_UPDATER_H
#define CE_UPDATER_H
#include <vector>
#include <string>
#include <map>
#include "matrix.hpp"
#include <Python.h>

typedef std::vector< std::vector<std::string> > name_list;
typedef std::vector< std::vector< std::vector<double> > > cluster_list;
typedef std::map<std::string,double> bf_list;
typedef std::map<std::string,double> cf;

enum class Status_t {
  READY, INIT_FAILED
};

class CEUpdater
{
public:
  CEUpdater( PyObject *BC, PyObject *corrFunc );
private:
  void create_ctype_lookup();

  std::vector<std::string> symbols;
  name_list cluster_names;
  cluster_list cluster_indx;
  bf_list basis_functions;
  cf* corr_functions{nullptr};
  Status_t status{Status_t::READY};
  Matrix trans_matrix;
  std::map<std::string,int> ctype_lookup;
};
#endif
