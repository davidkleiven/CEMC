#include "adaptive_timestep_logger.hpp"
#include <fstream>


using namespace std;

AdaptiveTimeStepLogger::AdaptiveTimeStepLogger(const string &fname): fname(fname){
    logfile.open(fname, ios::out | ios::app);

    if (!logfile.good()){
        throw runtime_error("Could not open logfile!");
    }

    if (!hasContent()){
        logfile << "# Step no., Time\n";
    }
}

AdaptiveTimeStepLogger::~AdaptiveTimeStepLogger(){
    logfile.close();
}

void AdaptiveTimeStepLogger::log(unsigned int iter, double t){
    logfile << iter << ", " << t << "\n";
    logfile.flush();
}

bool AdaptiveTimeStepLogger::hasContent(){
    logfile.seekp(0, ios::end);
    return logfile.tellp() != 0;
}

LogFileEntry AdaptiveTimeStepLogger::getLast() const{

    ifstream read_stream(fname);

    if (!read_stream.good()){
        throw runtime_error("Could not open logfile for reading!");
    }

    LogFileEntry entry;

    string next_line;
    string line;
    while(getline(read_stream, next_line)){
        line = next_line;
    }
    read_stream.close();

    try{
        // Split line
        unsigned int delim = line.find(",");
        entry.iter = stoi(line.substr(0, delim));
        entry.time = stod(line.substr(delim + 1));
    }
    catch (exception &e){
        entry.iter = 0;
        entry.time = 0.0;
    }
    return entry;
}