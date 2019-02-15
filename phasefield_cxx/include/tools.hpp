#ifndef PHASEFIELD_TOOLS_H
#define PHASEFIELD_TOOLS_H
#include "mmsp.grid.h"
#include "mmsp.vector.h"

template<int dim, typename T>
T partial_double_derivative(const MMSP::grid<dim, T> &GRID, const MMSP::vector<int> &x, unsigned int dir){
    MMSP::vector<int> s = x;
    const T& y = GRID(x);
    s[dir] += 1;
    const T& yh = GRID(s);
    s[i] -= 2;
    const T& yl = GRID(s);
    s[i] += 1;

    double weight = 1.0/pow(dx(GRID, dir), 2.0);
    return weight*(yh - 2.0*y + yl);
}
#endif