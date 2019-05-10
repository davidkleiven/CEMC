#ifndef ADAPTIVE_TIME_STEP_LOGGER_H
#define ADAPTIVE_TIME_STEP_LOGGER_H
#include <string>
#include <fstream>

struct LogFileEntry{
    unsigned int iter;
    double time;
};

class AdaptiveTimeStepLogger{
public:
    AdaptiveTimeStepLogger(const std::string &fname);
    ~AdaptiveTimeStepLogger();

    /** Log an entry */
    void log(unsigned int iter, double time);

    /** Read the last time step from the logfile */
    LogFileEntry getLast() const;
private:
    std::string fname;
    std::ofstream logfile;

    bool hasContent();
};
#endif