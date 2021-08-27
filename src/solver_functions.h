#define GAMMA 1.4
#define GM1 0.4
#define GGM1 0.56

FVSAND_GPU_GLOBAL
void init_q(double *q, double *center, double *flovar, int nfields, int istor, int ncells)
{
  int scale=(istor==0)?nfields:1;
  int stride=(istor==0)?1:ncells;
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells) 
#else
  for(int idx=0;idx<ncells;idx++)
#endif
    {
      double q0[5];
      q0[0]=flovar[0];
      q0[1]=flovar[0]*flovar[1]+(center[3*idx])*0.1;
      q0[2]=flovar[0]*flovar[2]+(center[3*idx+1]+center[3*idx]*center[3*idx]+center[3*idx+2])*0.1;
      q0[3]=flovar[0]*flovar[3]+(center[3*idx+2])*0.1;
      q0[4]=flovar[4]/GM1 + 0.5*(q0[1]*q0[1]+q0[2]*q0[2]+q0[3]*q0[3])/q0[0];
      for(int n=0;n<nfields;n++)
	q[idx*scale+n*stride]=q0[n];
    }
}
