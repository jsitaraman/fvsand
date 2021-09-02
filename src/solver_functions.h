#define GAMMA 1.4
#define GM1 0.4
#define GGM1 0.56
#include <stdio.h>
#include "roe_flux3d.h"
#include "mathops.h"

FVSAND_GPU_GLOBAL
void init_q(double *q0, double *q, double *dq,  double *center, double *flovar, int nfields, int istor, int ncells)
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
      for(int n=0;n<nfields;n++){
	q[idx*scale+n*stride]=q0[n];
        dq[idx*scale+n*stride] = 0.0; 
      }
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
void jacobiSweep(double *res, double *dq, double *normals,double *volume,
		 double *flovar, double *faceq, double *face_norm, int *cell2cell, int *cell2face, int *nccft, int nfields, int istor, int ncells, 
		 int* facetype, double dt, int iter)
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
	double dqtemp[5];
 	double B[5], Btmp[5];
	double D[25]; 
        double lmat[25]; 
        double rmat[25]; 
	int index1; 

	for(int n = 0; n<nfields; n++) {
		if(iter==0){
			dqtemp[n] = 0.0; 
		}
		else{
			dqtemp[n] = dq[scale*idx+n*stride]; 
		}
		B[n] = -res[scale*idx+n*stride]; 
		for(int m = 0; m<nfields; m++) {
			index1 = n*nfields + m;
			if(n==m){
				D[index1] = volume[idx]/dt;
			}
			else{
				D[index1] = 0.0; 
			}
		}
	}
 	// Loop over neighbors
      	for(int f=nccft[idx];f<nccft[idx+1];f++){
		double *norm=normals+18*idx+3*(f-nccft[idx]);
		int faceid=cell2face[f];
	       	faceid=abs(faceid)-1;

  		double* ql=faceq+(2*faceid)*nfields;
		double* qr=faceq+(2*faceid+1)*nfields;
		norm=face_norm+faceid*3;

		for(int n = 0; n<nfields; n++){
		       Btmp[n] =0.0;
      		       if (facetype[faceid] == -3) qr[n]=flovar[n];
		}

		//Compute Jacobians 
		computeJacobian(ql[0], ql[1],  ql[2],  ql[3],  ql[4],
	                        qr[0], qr[1],  qr[2],  qr[3],  qr[4],  
         	                norm[0], norm[1], norm [2],
                  	        faceid,lmat, rmat);

		for(int n = 0; n<5; n++){
		for(int m = 0; m<5; m++){
			index1 = 5*n+m;
 if(idx==0) printf("idx 0:n = %i, m = %i, lmat[ind1] = %f,rmat[ind1] = %f\n",n,m,lmat[index1],rmat[index1]); 
		}
		}
		//Compute Di and Oij dq
		axb1(rmat,dqtemp,Btmp,1,5); 
		for(int n = 0; n<5; n++){
			B[n] = B[n] - Btmp[n]; // XXX why doesn't this work?
 if(idx==0) printf("idx 0:n = %i, B = %f, Btmp = %f\n",n,B[n],Btmp[n]);
			for(int m = 0; m<5; m++){
				index1 = n*5+m; 
				D[index1] = D[index1] + lmat[index1];
 if(idx==0) printf("idx 0:n = %i,m = %i, D = %f\n",n,m,D[index1]);
			}
		}
	}

	// Compute dqtilde
	invertMat5(D,B,dqtemp); //computes dqtemp = inv(D)*B

	for(int n=0;n<nfields;n++){
		dq[scale*idx+n*stride] = dqtemp[n]; 
	}
  } // loop over cells 
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


FVSAND_GPU_GLOBAL
void fill_faces(double *q, double *faceq, int *nccft,int *cell2face,
		int nfields, int istor, int ncells)
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
      for(int f=nccft[idx];f<nccft[idx+1];f++)
	{
	  int faceid=cell2face[f];
	  int isgn=abs(faceid)/faceid;
	  int offset=(1-isgn)*nfields/2;
	  faceid=abs(faceid)-1;
	  for(int n=0;n<nfields;n++)
	    faceq[2*faceid*nfields+n+offset]=q[scale*idx+n*stride];
	}      
    }
}
FVSAND_GPU_GLOBAL
void face_flux(double *faceflux,double *faceq, double *face_norm, double *flovar,
	       int *facetype,
	       int nfields,int nfaces)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nfaces)
#else
  for(int idx=0;idx<nfaces;idx++)
#endif
    {
      double *ql=faceq+(2*idx)*nfields;
      double *qr=faceq+(2*idx+1)*nfields;
      double *dres=faceflux+idx*nfields;
      double *norm=face_norm+idx*3;
      double gx,gy,gz;
      gx=gy=gz=0;
      double spec;
      int idxn=facetype[idx];
      if (idxn == -3) {
       for(int n=0;n<nfields;n++)
          qr[n]=flovar[n];
      }
      InterfaceFlux_Inviscid(dres[0],dres[1],dres[2],dres[3],dres[4],
			     ql[0],ql[1],ql[2],ql[3],ql[4],
			     qr[0],qr[1],qr[2],qr[3],qr[4],
			     norm[0],norm[1],norm[2],
			     gx,gy,gz,spec,idxn);      
    }
}

//res_d,faceflux_d,volume_d,cell2face_d,nccft_d,
//                         nfields_d,istor,ncells);

FVSAND_GPU_GLOBAL
void computeResidualFace(double *res, double *faceflux, double *volume,
			 int *cell2face, int *nccft, int nfields,
			 int istor, int ncells)
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
	  int faceid=cell2face[f];
	  int isgn=faceid/abs(faceid);
	  faceid=abs(faceid)-1;
	  for(int n=0;n<nfields;n++)
	    res[scale*idx+n*stride]-=(isgn*faceflux[faceid*nfields+n]);
	}
      for(int n=0;n<nfields;n++) res[scale*idx+n*stride]/=volume[idx];
    }
}
