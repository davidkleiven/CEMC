#ifndef PAIR_CONSTRAINT_H
#define PAIR_CONSTRAINT_H
#include <string>
#include <vector>
#include <Python.h>

class CEUpdater;
class SymbolChange;

class PairConstraint
{
public:
  PairConstraint(const CEUpdater &updater, const std::string &cluster_name, \
    const std::string &elemt1, const std::string &elem2);

  /** Checks if a pair of the two elements exists */
  bool elems_in_pair( const std::vector<SymbolChange> &changes ) const;
  bool elems_in_pair( PyObject *system_changes ) const;
private:
  const CEUpdater *updater{nullptr};
  std::string cluster_name;
  std::string elem1;
  std::string elem2;

  /** Update the symbols array */
  void update_symbols( const std::vector<SymbolChange> &changes, std::vector<std::string> &symbols ) const;
};
#endif
