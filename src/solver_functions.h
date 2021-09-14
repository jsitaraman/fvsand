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

// Verify compute Jacobian routine is working correctly
// Make sure that F{qr+dqr,ql+dql) - F{qr,ql} = dql*lmat + dqr*rmat
FVSAND_GPU_GLOBAL void testComputeJ(double *q, double *normals,
				    double *flovar, int *cell2cell, int *nccft, int nfields, int istor, int ncells,int* facetype)
{
  double lmat[25], rmat[25];
  double dql[5],dqr[5],lmatdql[5], rmatdqr[5];
  double rhs[5],lhs[5];
  double ql[5],qr[5];
  double gx,gy,gz;

  int scale=(istor==0)?nfields:1;
  int stride=(istor==0)?1:ncells;

  // Setup arbitrary inputs
  int idx = 12230; // arbritrary 
  int f = 2;	  // arbritrary

  for(int n = 0; n<nfields; n++){
    //        ql[n]= n+1;
    //        qr[n] = (n+1)*1.1;
    dql[n] = .001;
    dqr[n] = .002;
    rmatdqr[n] = 0; 
    lmatdql[n] = 0; 
  }
  /*  double norm[3];
      norm[1] = 0.19245;
      norm[2] = 0.96225;
      norm[3] = -0.19245;
  */
  gx=gy=gz=0;

  int idxn=cell2cell[f];
  for(int n=0;n<nfields;n++) ql[n]=q[scale*idx+n*stride];
  if (idxn > -1) for(int n=0;n<nfields;n++) qr[n]=q[scale*idxn+n*stride];
  if (idxn == -3) for(int n=0;n<nfields;n++) qr[n]=flovar[n];
  double *norm=normals+18*idx+3*(f-nccft[idx]);

  
  printf("\nInputs:\n==========\n");
  for(int n=0;n<3;n++)	  printf("norm[%i] = %e\n",n,norm[n]);
  for(int n=0;n<5;n++)	  printf("ql[%i] = %e\n",n,ql[n]);
  for(int n=0;n<5;n++)	  printf("qr[%i] = %e\n",n,qr[n]);
  for(int n=0;n<5;n++)	  printf("dql[%i] = %e\n",n,dql[n]);
  for(int n=0;n<5;n++)	  printf("dqr[%i] = %e\n",n,dqr[n]);
  
  //RHS: Compute Jacobians 
  //  idxn = facetype[idx];
  printf("idxn = %i\n",idxn);
  computeJacobian(ql[0], ql[1],  ql[2],  ql[3],  ql[4],
                  qr[0], qr[1],  qr[2],  qr[3],  qr[4],
                  norm[0], norm[1], norm [2],
                  idxn,lmat, rmat);
  axb1(rmat,dqr,rmatdqr,1,5);
  axb1(lmat,dql,lmatdql,1,5);
  printf("\nOutputs:\n==========\n");
  for(int n=0;n<5;n++) for(int m =0;m<5;m++)	  printf("rmat[%i] = %e\n",n*5+m,rmat[n*5+m]);
  for(int n=0;n<5;n++) for(int m =0;m<5;m++)	  printf("lmat[%i] = %e\n",n*5+m,lmat[n*5+m]);
  for(int n=0;n<5;n++) printf("rmatdqr[%i] = %e\n",n,rmatdqr[n]);
  for(int n=0;n<5;n++) printf("lmatdql[%i] = %e\n",n,lmatdql[n]);

  for(int n=0;n<5;n++) rhs[n] = rmatdqr[n]+lmatdql[n];
  
  //LHS: Compute fluxes
  double flux[5], flux2[5],spec;
  for(int n =0; n<5;n++) {
    flux[n] = 0.0;
    flux2[n] = 0.0; 
  }
  InterfaceFlux_Inviscid(flux[0],flux[1],flux[2],flux[3],flux[4],
                         ql[0],ql[1],ql[2],ql[3],ql[4],
                         qr[0],qr[1],qr[2],qr[3],qr[4],
                         norm[0],norm[1],norm[2],
                         gx,gy,gz,spec,idxn);
  for(int n =0; n<5;n++) {
    ql[n] = ql[n]+dql[n];
    qr[n] = qr[n]+dqr[n];
  }
  InterfaceFlux_Inviscid(flux2[0],flux2[1],flux2[2],flux2[3],flux2[4],
                         ql[0],ql[1],ql[2],ql[3],ql[4],
                         qr[0],qr[1],qr[2],qr[3],qr[4],
                         norm[0],norm[1],norm[2],
                         gx,gy,gz,spec,idxn);
  for(int n =0; n<5;n++) lhs[n] = flux2[n]-flux[n];
  for(int n=0;n<5;n++) printf("flux0[%i] = %e\n",n,flux[n]);
  for(int n=0;n<5;n++) printf("flux2[%i] = %e\n",n,flux2[n]);

  for(int n=0;n<5;n++) printf("DEBUG VERIFY: RHS[%i] = %e, LHS = %e, DIFF = %e\n",n,rhs[n],lhs[n],rhs[n]-lhs[n]);

}
	

FVSAND_GPU_GLOBAL
void jacobiSweep(double *q, double *res, double *dq, double *dqupdate, double *normals,double *volume,
		 double *flovar, int *cell2cell, int *nccft, int nfields, int istor, int ncells, 
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
	double dqtemp[5],dqn[5];
 	double B[5], Btmp[5];
	double D[25]; 
        double lmat[25]; 
        double rmat[25]; 
	int index1; 

	for(int n = 0; n<nfields; n++) {
	  dqtemp[n] = dq[scale*idx+n*stride]; 
	  B[n] = res[scale*idx+n*stride]; 
	  for(int m = 0; m<nfields; m++) {
	    index1 = n*nfields + m;
	    if(n==m){
	      D[index1] = 1.0/dt;
	    }
	    else{
	      D[index1] = 0.0; 
	    }
	  }
	}
 	// Loop over neighbors
        for(int f=nccft[idx];f<nccft[idx+1];f++)
	  {
	    double *norm=normals+18*idx+3*(f-nccft[idx]);
	    int idxn=cell2cell[f];
	    double ql[5],qr[5];
	    for(int n=0;n<nfields;n++) {
	      ql[n]=q[scale*idx+n*stride];
	      Btmp[n] =0.0;
	    }
	    if (idxn > -1) {
	      for(int n=0;n<nfields;n++) qr[n]=q[scale*idxn+n*stride];
	    }
	    if (idxn == -3) {
	      for(int n=0;n<nfields;n++) qr[n]=flovar[n];
	    }
	
	    //Compute Jacobians 
	    computeJacobian(ql[0], ql[1],  ql[2],  ql[3],  ql[4],
			    qr[0], qr[1],  qr[2],  qr[3],  qr[4],  
			    norm[0], norm[1], norm [2],
			    idxn,lmat, rmat);

	    for(int n = 0; n<nfields; n++){
	      for(int m = 0; m<nfields; m++){
		index1 = n*nfields + m; 
		lmat[index1] /= volume[idx];
		rmat[index1] /= volume[idx];
	      }
	    }
	    //Compute Di and Oij*dq_neighbor
	    for(int n=0; n<5; n++) {
	      if (idxn > -1) {
		dqn[n] = dq[scale*idxn+n*stride];
	      }
	      else {
		dqn[n] = 0.0;
	      }
	    }
	    axb1(rmat,dqn,Btmp,1,5); 
	    for(int n = 0; n<5; n++){
	      B[n] = B[n] - Btmp[n]; 
	      for(int m = 0; m<5; m++){
		index1 = n*5+m; 
		D[index1] = D[index1] + lmat[index1];
	      }
	    }
	  }
	// Compute dqtilde and send back out of kernel
	solveAxb5(D,B,dqtemp); // compute dqtemp = inv(D)*B
	for(int n=0;n<nfields;n++) dqupdate[scale*idx+n*stride] = dqtemp[n]; 
      } // loop over cells 
}

FVSAND_GPU_GLOBAL
void setValues(double *qdest, double qsrc, int ndof)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ndof) 
#else
    for(int idx=0;idx<ndof;idx++)
#endif
      {
	qdest[idx]=qsrc;
      }
}

FVSAND_GPU_GLOBAL
void copyValues(double *qdest, double *qsrc, int ndof)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ndof) 
#else
    for(int idx=0;idx<ndof;idx++)
#endif
      {
	qdest[idx]=qsrc[idx];
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
