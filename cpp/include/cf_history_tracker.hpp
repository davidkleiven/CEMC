#ifndef CF_HISTORY_TRACKER_H
#define CF_HISTORY_TRACKER_H
#include <string>
#include <array>

typedef std::map<std::string,double> cf;

struct SystemChanges
{
  std::string old_symbol;
  std::string new_symbol;
  unsigned int indx;
};

class CFHistoryTracker
{
public:
  CFHistoryTracker();

  /** Return a pointer to the next pointer to be written to */
  cf* get_next();
private:
  static const unsigned int max_history = 10;
  std::array<cf,max_history> cf_hist;
  std::array<SystemChanges,max_history> changes;
  unsigned int current{0};
};
#endif
