#include <math.h>
void curl(double *A, double *B , double *result)
{
  result[0]=A[1]*B[2]-B[1]*A[2];
  result[1]=A[2]*B[0]-B[2]*A[0];
  result[2]=A[0]*B[1]-B[0]*A[1];
}

void normalize(double *vec)
{
  double vmag=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  vec[0]/=vmag;
  vec[1]/=vmag;
  vec[2]/=vmag;
}
