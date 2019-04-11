#include "chc_noise.hpp"
#include <cmath>


template<int dim>
CHCNoise<dim>::CHCNoise(double mobility, double dt, double amplitude, unsigned int L): mobility(mobility), dt(dt), amplitude(amplitude), L(L){
    switch (dim){
        case 1:
            indexGrid = new MMSP::grid<dim, int>(1, 0, L);
            break;
        case 2:
            indexGrid = new MMSP::grid<dim, int>(1, 0, L, 0, L);
            break;
        case 3:
            indexGrid = new MMSP::grid<dim, int>(1, 0, L, 0, L, 0, L);
    }

    for (int i=0;i<MMSP::nodes(*indexGrid);i++){
        (*indexGrid)(i) = i;
    }
}

template<int dim>
CHCNoise<dim>::~CHCNoise(){
    delete indexGrid; indexGrid = nullptr;
}

template<int dim>
void CHCNoise<dim>::create(std::vector<double> &noise) const{
    double pi = acos(-1.0);
    unsigned int N = MMSP::nodes(*indexGrid);
    noise.resize(N);
    vector<double> white_noise(3*N);

    if (N%2 != 0){
        throw runtime_error("Use an even number of grid points when using CHC noise!");
    }

    for (unsigned int i=0;i<3*N;i+=2){
        double rand1 = static_cast<double>(rand())/RAND_MAX;
        double rand2 = static_cast<double>(rand())/RAND_MAX;

        double normal1 = sqrt(-2.0*log(rand1))*cos(2*pi*rand2);
        double normal2 = sqrt(-2.0*log(rand1)*sin(2*pi*rand2));
        white_noise[2*i] = normal1;
        white_noise[2*i+1] = normal2;
    }

     
}

template<int dim>
void CHCNoise<dim>::chc_noise(const std::vector<double> &white_noise, std::vector<double> &noise) const{
    for (unsigned int i=0;i<MMSP::nodes(*indexGrid);i++){
        MMSP::vector<int> pos = indexGrid->position(i);

        double divergence = 0.0;
        for (unsigned int dir=0;dir < dim;dir++){
            int orig_pos = pos[dir];
            pos[dir] = (pos[dir] + 1)%L;
            unsigned int indx = (*indexGrid)(pos);

            divergence += (white_noise[3*indx + dir] - white_noise[3*i + dir]);
            pos[dir] = orig_pos;
        }

        noise[i] = sqrt(amplitude*mobility/dt)*divergence;
    }
}

template class CHCNoise<1>;
template class CHCNoise<2>;
template class CHCNoise<3>;