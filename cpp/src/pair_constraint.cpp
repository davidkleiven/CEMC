#include "pair_constraint.hpp"
#include "cluster.hpp"
#include "row_sparse_struct_matrix.hpp"
#include "ce_updater.hpp"
#include <map>
#include "cf_history_tracker.hpp"

using namespace std;

PairConstraint::PairConstraint(const CEUpdater &updater, const string &cluster_name, \
  const string& elem1, const string& elem2):updater(&updater), cluster_name(cluster_name), \
  elem1(elem1), elem2(elem2){};

bool PairConstraint::elems_in_pair( const vector<SymbolChange> &changes ) const
{
  // Get a copy of the symbols array
  vector<string> symbs = updater->get_symbols();
  update_symbols( changes, symbs );

  map<unsigned int, const Cluster*> clusters;
  updater->get_clusters(cluster_name, clusters);
  const RowSparseStructMatrix& trans_mat = updater->get_trans_matrix();

  for (unsigned int i=0;i<symbs.size();i++ )
  {
    if (symbs[i] == elem1)
    {
      unsigned int symm_group = updater->get_trans_symm_group(i);

      auto iter = clusters.find(symm_group);

      if ( iter != clusters.end() )
      {
        const vector< vector<int> >& members = iter->second->get();
        for (unsigned int j=0;j<members.size();j++)
        {
          int indx = trans_mat(i,members[j][0]);
          if (symbs[indx] == elem2)
          {
            return true;
          }
        }
      }
    }
  }
  return false;
}


bool PairConstraint::elems_in_pair( PyObject *system_changes ) const
{
  vector<SymbolChange> changes;
  updater->py_changes2_symb_changes(system_changes, changes);
  return elems_in_pair(changes);
}


void PairConstraint::update_symbols( const vector<SymbolChange> &changes, vector<string> &symbols ) const
{
  for (auto iter=changes.begin(); iter != changes.end(); ++iter )
  {
    symbols[iter->indx] = iter->new_symb;
  }
}
