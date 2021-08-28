#include<math.h>

//
// number of vertices per face for all regular polyhedra and
// their implicit connectivity, using Dimitri's mcell style description
// http://scientific-sims.com/oldsite/index.php/solver-file-formats/cell-type-definitions-dm
//
// TODO
// This is repeated from GlobalMesh (find a common container in implementation that is
// synced across both cpu and gpu)
// and also repeated in f90
//
FVSAND_GPU_DEVICE
int numverts[4][6]={3,3,3,3,0,0,4,3,3,3,3,0,3,4,4,4,3,0,4,4,4,4,4,4};
FVSAND_GPU_DEVICE
int face2node[4][24]={1,2,3,3,1,4,2,2,2,4,3,3,1,3,4,4,0,0,0,0,0,0,0,0,
		      1,2,3,4,1,5,2,2,2,5,3,3,4,3,5,5,1,4,5,5,0,0,0,0,
		      1,2,3,3,1,4,5,2,2,5,6,3,1,3,6,4,4,6,5,5,0,0,0,0,
		      1,2,3,4,1,5,6,2,2,6,7,3,3,7,8,4,1,4,8,5,5,8,7,6};

FVSAND_GPU_DEVICE
double scalarTripleProduct(double a[3],double b[3],double c[3])
{
  double sp;
  sp= a[0]*b[1]*c[2] - a[0]*b[2]*c[1]
    +a[1]*b[2]*c[0] - a[1]*b[0]*c[2]
    +a[2]*b[0]*c[1] - a[2]*b[1]*c[0];
  return sp;
}

FVSAND_GPU_DEVICE
void curlp(double *A, double *B , double *result)
{
  result[0]+=(A[1]*B[2]-B[1]*A[2]);
  result[1]+=(A[2]*B[0]-B[2]*A[0]);
  result[2]+=(A[0]*B[1]-B[0]*A[1]);
}

// not used now
FVSAND_GPU_DEVICE
double computeCellVolume(double xv[8][3],int nvert)
{
  double vol;
  int itype;
  int nfaces;
  int numverts[4][6]={3,3,3,3,0,0,4,3,3,3,3,0,3,4,4,4,3,0,4,4,4,4,4,4};
  int faceInfo[4][24]={1,2,3,0,1,4,2,0,2,4,3,0,1,3,4,0,0,0,0,0,0,0,0,0,
                       1,2,3,4,1,5,2,0,2,5,3,0,4,3,5,0,1,4,5,0,0,0,0,0,
                       1,2,3,0,1,4,5,2,2,5,6,3,1,3,6,4,4,6,5,0,0,0,0,0,
                       1,2,3,4,1,5,6,2,2,6,7,3,3,7,8,4,1,4,8,5,5,8,7,6};
 switch(nvert)
   {
   case 4:
     itype=0;
     nfaces=4;
     break;
   case 5:
     itype=1;
     nfaces=5;
     break;
   case 6:
     itype=2;
     nfaces=5;
     break;
   case 8:
     itype=3;
     nfaces=6;
     break;
   }

 vol=0.0;
 for(int iface=0;iface<nfaces;iface++)
   {
     if (numverts[itype][iface]==3) {
       vol-=0.5*scalarTripleProduct(xv[faceInfo[itype][4*iface+0]-1],
			      xv[faceInfo[itype][4*iface+1]-1],
			      xv[faceInfo[itype][4*iface+2]-1]);
     } else {
       vol-=0.25*scalarTripleProduct(xv[faceInfo[itype][4*iface+0]-1],
			       xv[faceInfo[itype][4*iface+1]-1],
			       xv[faceInfo[itype][4*iface+2]-1]);
       vol-=0.25*scalarTripleProduct(xv[faceInfo[itype][4*iface+0]-1],
			       xv[faceInfo[itype][4*iface+2]-1],
			       xv[faceInfo[itype][4*iface+3]-1]);
       vol-=0.25*scalarTripleProduct(xv[faceInfo[itype][4*iface+0]-1],
			       xv[faceInfo[itype][4*iface+1]-1],
			       xv[faceInfo[itype][4*iface+3]-1]);
       vol-=0.25*scalarTripleProduct(xv[faceInfo[itype][4*iface+1]-1],
			       xv[faceInfo[itype][4*iface+2]-1],
			       xv[faceInfo[itype][4*iface+3]-1]);
     }
   }
 vol/=3.0;
 return vol;
}


// need to change this routine to cell centroid instead of
// geometric center

FVSAND_GPU_GLOBAL
void cell_center(double *center,double *x,int *nvcft,int *cell2node,int ncells)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells*3) 
#else
  for(int idx=0;idx<ncells*3;idx++)
#endif
    {
      int i=idx/3;
      int n=idx%3;
      center[idx]=0;
      for(int j=nvcft[i];j<nvcft[i+1];j++)
	center[idx]+=(x[3*cell2node[j]+n]);
      double scal=1./(nvcft[i+1]-nvcft[i]);
      center[idx]*=scal;
    }  
}

FVSAND_GPU_GLOBAL
void cell_normals_volume(double *normals,
			 double *volume,
			 double *x,
			 int *ncon,
			 int *cell2node,
			 int *nvcft,
			 int ncells)
{

#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells) 
#else
  double normsum[3];
  double totalvol=0;
  normsum[0]=normsum[1]=normsum[2]=0;
  for(int idx=0;idx<ncells;idx++)
#endif
    {
      int m=0;
      int v[8];
      double xv[8][3];
      for(int j=nvcft[idx];j<nvcft[idx+1];j++)
	v[m++]=cell2node[j];
      /* 	{ */
      /* 	  v[m]=cell2node[j]; */
      /* 	  xv[m][0]=x[3*v[m]]; */
      /* 	  xv[m][1]=x[3*v[m]+1]; */
      /* 	  xv[m][2]=x[3*v[m]+2]; */
      /* 	  m++; */
      /* 	} */
      /* volume[idx]=computeCellVolume(xv,nvcft[idx+1]-nvcft[idx]); */
      // kludge to convert [4,5,6,8] to [0,1,2,3]
      int itype=(nvcft[idx+1]-nvcft[idx]-4);
      itype=(itype > 3) ? 3: itype;
      
      double vol=0.0;
      for(int f=0;f<ncon[idx];f++)
	{
          double xf[12],A[3],B[3];
	  for(int p=0;p<numverts[itype][f];p++)
	    {
	      int i=face2node[itype][4*f+p]-1;
	      xf[3*p  ]=x[3*v[i]];
	      xf[3*p+1]=x[3*v[i]+1];
	      xf[3*p+2]=x[3*v[i]+2];
	    }
	  // pointer to the right normal
	  // norm is not unit normal
	  double *norm=normals+18*idx+3*f;
	  norm[0]=norm[1]=norm[2]=0;
	  //
	  // unified loop for triangles and quads
	  // without an explicit if loop
	  // for quads both set of diagonal division to
	  // triangle are processed and added together
	  //
	  //
	  double vscal=1./(numverts[itype][f]-2);
	  for(int v0=0;v0<numverts[itype][f];v0+=3)
	    {
	      for(int p=0;p<numverts[itype][f]-2;p++)
		{
		  int p1=(v0+p+1)%4;
		  int p2=(v0+p+2)%4;
		  A[0]=xf[3*p1  ]-xf[3*v0];
		  A[1]=xf[3*p1+1]-xf[3*v0+1];
		  A[2]=xf[3*p1+2]-xf[3*v0+2];
		  
		  B[0]=xf[3*p2  ]-xf[3*v0];
		  B[1]=xf[3*p2+1]-xf[3*v0+1];
		  B[2]=xf[3*p2+2]-xf[3*v0+2];	      
		  // remember the face connectivity is 
		  // facing towards the interior of the cell
		  curlp(B,A,norm);
		  // compute cell volume using Gauss-Divergence formulae
		  vol-=(vscal*0.5*scalarTripleProduct(&(xf[3*v0]),&(xf[3*p1]),&(xf[3*p2])));
		}	      
	    }

	  norm[0]*=vscal;
	  norm[1]*=vscal;
	  norm[2]*=vscal;
	}
      //printf("%lf %lf\n",volume[idx],vol/3.0);
      volume[idx]=(vol/3.0);
      totalvol+=volume[idx];
      for(int f=0;f<ncon[idx];f++)
	{
	  double *norm=normals+18*idx+3*f;
	  normsum[0]+=norm[0];
	  normsum[1]+=norm[1];
	  normsum[2]+=norm[2];
	}
    }
  printf("normalsum: %lf %lf %lf %lf\n",normsum[0],normsum[1],normsum[2],totalvol);
}    

FVSAND_GPU_GLOBAL
void check_conservation(double *normals,
			double *x,
			int *nccft,
			int *cell2cell,
			int ncells)
{
  //char fn[20];
  //sprintf(fn,"cons%d.dat",ncells);
  //FILE *fp=fopen(fn,"w");
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells)
#else
  double conscheck=0.0;
  for(int idx=0;idx<ncells;idx++)    
#endif
    {
      for(int f=nccft[idx];f<nccft[idx+1];f++)
	{
	  int idxn=cell2cell[f];
	  if (idxn > -1 && idxn < ncells) {
	    int f1;
	    for(f1=nccft[idxn];f1<nccft[idxn+1];f1++)
	      if ( cell2cell[f1]==idx) break;
	    double *norm=normals+18*idx+3*(f-nccft[idx]);
	    double *norm1=normals+18*idxn+3*(f1-nccft[idxn]);
	    //fprintf(fp,"%lf %lf %lf %lf %lf %lf\n",norm[0],norm[1],norm[2],norm1[0],norm1[1],norm1[2]);
	    conscheck+=(fabs(norm[0]+norm1[0])+
			fabs(norm[1]+norm1[1])+
			fabs(norm[2]+norm1[2]));
	  }
	}
    }
  printf("conscheck=%f\n",conscheck);
  //fclose(fp);
}
  
