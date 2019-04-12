#include "chc_noise.hpp"
#include <cmath>
#include <ctime>

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

    srand(time(0));
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
    vector<double> white_noise(dim*N);

    if (N%2 != 0){
        throw runtime_error("Use an even number of grid points when using CHC noise!");
    }

    for (unsigned int i=0;i<dim*N/2;i++){
        double rand1 = static_cast<double>(rand())/RAND_MAX;
        double rand2 = static_cast<double>(rand())/RAND_MAX;

        double normal1 = sqrt(-2.0*log(rand1))*cos(2*pi*rand2);
        double normal2 = sqrt(-2.0*log(rand1))*sin(2*pi*rand2);
        white_noise[2*i] = normal1;
        white_noise[2*i+1] = normal2;
    }

    chc_noise(white_noise, noise);
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

            divergence += (white_noise[dim*indx + dir] - white_noise[dim*i + dir]);
            pos[dir] = orig_pos;
        }
        noise[i] = sqrt(amplitude*mobility/dt)*divergence;
    }
}

template<int dim>
void CHCNoise<dim>::noise2grid(const std::string &fname, const std::vector<double> &noise) const{

    MMSP::grid<dim, double> grid(*indexGrid, 1);

    for (unsigned int i=0;i<noise.size();i++){
        grid(i) = noise[i];
    }

    grid.output(fname.c_str());
}

template class CHCNoise<1>;
template class CHCNoise<2>;
template class CHCNoise<3>;