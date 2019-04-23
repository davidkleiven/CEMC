#ifndef ORIGIN_SINGULARITY_INTEGRATION_H
#define ORIGIN_SINGULARITY_INTEGRATION_H

#include "MMSP.vector.h"

class OriginSingularityIntegration{
public:
    OriginSingularityIntegration(unsigned int N): N(N){};

    void get_integration_directions(unsigned int dim, std::vector< MMSP::vector<double> > &directions) const;
private:
    unsigned int N{2};

    void get_dir1D(std::vector< MMSP::vector<double> > &directions) const;
    void get_dir2D(std::vector< MMSP::vector<double> > &directions) const;
    void get_dir3D(std::vector< MMSP::vector<double> > &directions) const;
};
#endif