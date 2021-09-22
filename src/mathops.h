#include <stdio.h>
//#define UNIT_CHECK
#define REAL double

FVSAND_GPU_DEVICE void solveAxb5(double *A, double *b, double* x)
{
	double s; 
	int index1, index2;
       	double L[25], U[25]; 
	for(int i=0; i<25; i++){
		L[i] = 0.0; 
		U[i] = 0.0; 
	}	
	// LU Decomp
        for (int j=0; j<5; j++) {
          for (int k=0; k<j; k++) {
            L[5*j+k] = A[5*j+k];
          }
          for (int k=1; k<j; k++) {
            for (int l=0; l<k; l++) {
              L[5*j+k] = L[5*j+k] - L[5*j+l]*U[5*l+k];
            }
          }

          L[5*j+j] = A[5*j+j];
          for (int k=0; k<j; k++) {
            L[5*j+j] = L[5*j+j] - L[5*j+k]*U[5*k+j];
          }
          L[5*j+j] = 1.0/L[5*j+j];

          for (int k=j+1; k<5; k++) {
            U[5*j+k] = A[5*j+k];
	    for (int l=0; l<j; l++) {
              U[5*j+k] = U[5*j+k] - L[5*j+l]*U[5*l+k];
            }
            U[5*j+k] = U[5*j+k]*L[5*j+j];
          }
        }

	// Forward prop to solve Ly = b
	double y[5]; 	
        for (int j=0; j<5; j++) {
          y[j] = b[j];
          for (int k=0; k<j; k++) {
            y[j] = y[j] - L[5*j+k]*y[k];
          }
          y[j] = y[j]*L[5*j+j];
        }

	for (int j=4; j>-1; j--) {
          x[j] = y[j];
          for (int k=j+1; k<5; k++) {
            x[j] = x[j] - U[5*j+k]*x[k];
          }
        }
}


//
// perform x = inv (A)*f
// currently set for 5x5 matrix
//
//
FVSAND_GPU_DEVICE void invertMat5(double *A, double *f, double *x)
{
  REAL b11,b21,b22,b31,b32,b33,b41,b42,b43,b44,b51,b52,b53,b54,b55;
  REAL u12,u13,u14,u15,u23,u24,u25,u34,u35,u45;
  REAL d1,d2,d3,d4,d5;
  //
  // decompose A into L and U
  //
  //
  b11=1./A[0];
  u12=A[1]*b11;
  u13=A[2]*b11;
  u14=A[3]*b11;
  u15=A[4]*b11;
  b21=A[5];
  b22=1./(A[6]-b21*u12);
  u23=(A[7]-b21*u13)*b22;
  u24=(A[8]-b21*u14)*b22;
  u25=(A[9]-b21*u15)*b22;
  b31=A[10];
  b32=A[11]-b31*u12;
  b33=1./(A[12]-b31*u13-b32*u23);
  u34=(A[13]-b31*u14-b32*u24)*b33;
  u35=(A[14]-b31*u15-b32*u25)*b33;
  b41=A[15];
  b42=A[16]-b41*u12;
  b43=A[17]-b41*u13-b42*u23;
  b44=1./(A[18]-b41*u14-b42*u24-b43*u34);
  u45=(A[19]-b41*u15-b42*u25-b43*u35)*b44;
  b51=A[20];
  b52=A[21]-b51*u12;
  b53=A[22]-b51*u13-b52*u23;
  b54=A[23]-b51*u14-b52*u24-b53*u34;
  b55=1./(A[24]-b51*u15-b52*u25-b53*u35-b54*u45);
  //
  d1=f[0]*b11;
  d2=(f[1]-b21*d1)*b22;
  d3=(f[2]-b31*d1-b32*d2)*b33;
  d4=(f[3]-b41*d1-b42*d2-b43*d3)*b44;
  d5=(f[4]-b51*d1-b52*d2-b53*d3-b54*d4)*b55;
  //
  x[4]=d5;
  x[3]=d4-u45*d5;
  x[2]=d3-u34*x[3]-u35*d5;
  x[1]=d2-u23*x[2]-u24*x[3]-u25*d5;
  x[0]=d1-u12*x[1]-u13*x[2]-u14*x[3]-u15*d5;
}


//
// perform b=fac*A*x
//
FVSAND_GPU_DEVICE
void axb1(double A[25],double *x,double *b,double fac,int N)
{
  int j,k;
  int index1; 
  for(j=0;j<N;j++)
  {
    b[j]=0;
    for(k=0;k<N;k++){
	    index1 = j*N+k;
	    b[j]+=(fac*A[index1]*x[k]);
    }
  }
}

FVSAND_GPU_DEVICE
void axb1s(double A[25],double *x,double *b,double fac,int N)
{
  int j,k;
  int index1;
  for(j=0;j<N;j++)
  {
    for(k=0;k<N;k++){
            index1 = j*N+k;
            b[j]-=(fac*A[index1]*x[k]);
    }
  }
}

#ifdef UNIT_CHECK
int main()
{
  REAL A5mat[5][5];
  REAL A4mat[4][4];
  REAL b5mat[5];
  REAL b4mat[4];
  REAL x5mat[5];
  REAL x4mat[4];

  int i,j;
  int ip1,im1;

  for(i=0;i<5;i++)
    for(j=0;j<5;j++)
      A5mat[i][j]=0;
  
  for(i=0;i<5;i++)
    {
      ip1=(i+1)%5;
      im1=(i==0)?4:i-1;
      A5mat[i][i]=-2+0.1*i;
      A5mat[ip1][i]=0.9;
      A5mat[im1][i]=1;
      b5mat[i]=1;
    }

  invertMat5(A5mat,b5mat,x5mat);

  for(i=0;i<5;i++)
    {
      for(j=0;j<5;j++)
	printf("%f ",A5mat[i][j]);
      printf("\n");
    }
  
  for(i=0;i<5;i++)
    printf("%f %f\n",b5mat[i],x5mat[i]);


  for(i=0;i<4;i++)
    for(j=0;j<4;j++)
      A4mat[i][j]=0;
  
  for(i=0;i<4;i++)
    {
      ip1=(i+1)%4;
      im1=(i==0)?3:i-1;
      A4mat[i][i]=-2;
      A4mat[ip1][i]=0.9;
      A4mat[im1][i]=1;
      b4mat[i]=1;
    }

  invertMat4(A4mat,b4mat,x4mat);

  for(i=0;i<4;i++)
    {
      for(j=0;j<4;j++)
	printf("%f ",A4mat[i][j]);
      printf("\n");
    }
  
  for(i=0;i<4;i++)
    printf("%f %f\n",b4mat[i],x4mat[i]);


}
#endif

  
  
