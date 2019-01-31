#include "linalg.hpp"


void inverse3x3(const mat3x3 &mat, mat3x3 &inv){
  // Set the inverse matrix equal to the identity matrix
	double determinant = 0;
	
	//finding determinant
	for(unsigned int i=0;i<3;i++)
		determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));
	
	for(unsigned int i=0;i<3;i++){
	for(unsigned int j=0;j<3;j++)
			inv[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
	}
}