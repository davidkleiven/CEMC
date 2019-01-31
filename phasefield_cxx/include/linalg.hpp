#ifndef LINALG_H
#define LINALG_H
#include <array>
typedef std::array<std::array<double,3>, 3> mat3x3;

/** Calculate the inverse of a 3x3 matrix */
void inverse3x3(const mat3x3 &inarray, mat3x3 &inv);
#endif