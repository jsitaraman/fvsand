#define GAMMA 1.4
#define GM1 0.4
#define GGM1 0.56
#include "roe_flux3d.h"

FVSAND_GPU_GLOBAL
void init_q(double *q0, double *q, double *center, double *flovar, int nfields, int istor, int ncells)
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
      q0[0]=flovar[0];
      q0[1]=flovar[0]*flovar[1]; //+(center[3*idx])*0.1;
      q0[2]=flovar[0]*flovar[2]; //+(center[3*idx+1]+center[3*idx]*center[3*idx]+center[3*idx+2])*0.1;
      q0[3]=flovar[0]*flovar[3]; //+(center[3*idx+2])*0.1;
      q0[4]=flovar[4]/GM1 + 0.5*(q0[1]*q0[1]+q0[2]*q0[2]+q0[3]*q0[3])/q0[0];
      for(int n=0;n<nfields;n++)
	q[idx*scale+n*stride]=q0[n];
    }
}


FVSAND_GPU_GLOBAL
void computeResidual(double *res, double *q, double *center, double *normals,double *volume,
		     double *flovar,int *cell2cell, int *nccft, int nfields, int istor, int ncells)
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
      for(int n=0;n<nfields;n++) res[scale*idx+n*stride]=0;
      for(int f=nccft[idx];f<nccft[idx+1];f++)
	{
	  double *norm=normals+18*idx+3*(f-nccft[idx]);
	  int idxn=cell2cell[f];
	  // first order now
	  double ql[5],qr[5];	  
	  for(int n=0;n<nfields;n++)
	      ql[n]=q[scale*idx+n*stride];
	  if (idxn > -1) {
	    for(int n=0;n<nfields;n++)
	      qr[n]=q[scale*idxn+n*stride];
	  }
	  if (idxn == -3) {
	    for(int n=0;n<nfields;n++)
	      qr[n]=flovar[n];
	  }
	  double dres[5];
	  double gx,gy,gz; // grid speeds
	  double spec;     // spectral radius
	  gx=gy=gz=0;
	  
	  InterfaceFlux_Inviscid(dres[0],dres[1],dres[2],dres[3],dres[4],
				 ql[0],ql[1],ql[2],ql[3],ql[4],
				 qr[0],qr[1],qr[2],qr[3],qr[4],
				 norm[0],norm[1],norm[2],
				 gx,gy,gz,spec,idxn);
	  for(int n=0;n<nfields;n++)
	    res[scale*idx+n*stride]-=dres[n];
	  
	}
      // divide by cell volume, this will have to move outside for
      // deforming grids
      for(int n=0;n<nfields;n++) res[scale*idx+n*stride]/=volume[idx];
    }
}

FVSAND_GPU_GLOBAL
void updateFields(double *res, double *qdest, double *qsrc, double fscal, int ndof)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ndof) 
#else
    for(int idx=0;idx<ndof;idx++)
#endif
      {
	qdest[idx]=qsrc[idx]+(fscal*res[idx]);
      }
}

FVSAND_GPU_GLOBAL
void updateHost(double *qbuf, double *q, int *d2h, int nupdate)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nupdate) 
#else
    for(int idx=0;idx<nupdate;idx++)
#endif
      {
	qbuf[idx]=q[d2h[idx]];
      }
}

FVSAND_GPU_GLOBAL
void updateDevice(double *q, double *qbuf, int *h2d, int nupdate)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nupdate) 
#else
    for(int idx=0;idx<nupdate;idx++)
#endif
      {
	q[h2d[idx]]=qbuf[idx];
      }
}
