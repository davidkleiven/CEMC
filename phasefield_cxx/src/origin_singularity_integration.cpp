#include "origin_singularity_integration.hpp"
#include <stdexcept>

using namespace std;

void OriginSingularityIntegration::get_dir1D(std::vector< MMSP::vector<double> > &directions) const{

    MMSP::vector<double> vec(3);
    vec[0] = 1.0;
    vec[1] = 0.0;
    vec[2] = 0.0;
    directions.push_back(vec);
    vec[0] = -1.0;
    directions.push_back(vec);
}

void OriginSingularityIntegration::get_dir2D(std::vector< MMSP::vector<double> > &directions) const{
    const double PI = acos(-1.0);
    for (unsigned int i=0;i<N;i++){
        double theta = 2.0*PI*i/N;
        MMSP::vector<double> vec(3);
        vec[0] = cos(theta);
        vec[1] = sin(theta);
        vec[2] = 0.0;
        directions.push_back(vec);
    }
}

void OriginSingularityIntegration::get_dir3D(std::vector< MMSP::vector<double> > &directions) const{
    const double PI = acos(-1.0);
    for (unsigned int i=0;i<N;i++)
    for (unsigned int j=0;j<N;j++){
        double phi = 2.0*PI*i/N;
        double theta = PI*i/N;

        MMSP::vector<double> vec(3);
        vec[0] = cos(phi)*sin(theta);
        vec[1] = sin(phi)*sin(theta);
        vec[2] = cos(theta);
        directions.push_back(vec);
    }
}

void OriginSingularityIntegration::get_integration_directions(unsigned int dim, std::vector< MMSP::vector<double> > &directions) const{
    switch(dim){
        case 1:
            get_dir1D(directions);
            break;
        case 2:
            get_dir2D(directions);
            break;
        case 3:
            get_dir3D(directions);
            break;
        default:
            throw invalid_argument("dim has to be 1, 2 or 3!");
    }
}