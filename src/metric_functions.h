#include<math.h>
#include<cstdio>
#define lsq_max(a,b) ((a) > (b)) ? (a) :(b)
#define KlsqMinVal 1e-12
#define KlsqCosFac 1.5

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
  result[0]=(A[1]*B[2]-B[1]*A[2]);
  result[1]=(A[2]*B[0]-B[2]*A[0]);
  result[2]=(A[0]*B[1]-B[0]*A[1]);
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
      int i=idx/ncells;
      int n=idx%ncells;
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
			 double *centroid,
			 double *face_centroid,
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
      double cx[3];
      cx[0]=cx[1]=cx[2]=0;
      //double xv[8][3];
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
      //for(int i=0;i<nvcft[idx+1]-nvcft[idx];i++)
      //	  printf("%f %f %f\n",x[3*v[i]],x[3*v[i]+1],x[3*v[i]+2]);
      //printf("\n");
      
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
	      //printf("%f,%f,%f\n",xf[3*p  ],xf[3*p+1],xf[3*p+2]);
	    }
	  // pointer to the right normal
	  // norm is not unit normal
          double norm[3];
	  norm[0]=norm[1]=norm[2]=0;
	  double fcx[3];
	  fcx[0]=fcx[1]=fcx[2]=0;
	  //
	  // unified loop for triangles and quads
	  // without an explicit if loop
	  // for quads both set of diagonal division to
	  // triangle are processed and added together
	  //
	  //
	  double vscal=0.5/(numverts[itype][f]-2);
	  double area=0;
	  for(int v0=0;v0<numverts[itype][f];v0+=3)
	    {
	      for(int p=0;p<numverts[itype][f]-2;p++)
		{
		  int p1=(v0+p+1)%4;
		  int p2=(v0+p+2)%4;
		  double nn[3];
		  A[0]=xf[3*p1  ]-xf[3*v0];
		  A[1]=xf[3*p1+1]-xf[3*v0+1];
		  A[2]=xf[3*p1+2]-xf[3*v0+2];
		  
		  B[0]=xf[3*p2  ]-xf[3*v0];
		  B[1]=xf[3*p2+1]-xf[3*v0+1];
		  B[2]=xf[3*p2+2]-xf[3*v0+2];	      
		  // remember the face connectivity is 
		  // facing towards the interior of the cell
		  curlp(B,A,nn);
		  norm[0]+=nn[0];
		  norm[1]+=nn[1];
		  norm[2]+=nn[2];
		  double darea=sqrt(nn[0]*nn[0]+nn[1]*nn[1]+nn[2]*nn[2])*vscal;
		  // compute cell volume using Gauss-Divergence formulae
		  //(1/3)*int_(x,y,z).n dS
		  vol-=(vscal*scalarTripleProduct(&(xf[3*v0]),&(xf[3*p1]),&(xf[3*p2])));
		  // compute cell centroid using divergence formula
		  // int_0.5*(x^2, 0, 0; 0, y^2, 0; 0 ,0, z^2).n dS
		  for(int d=0;d<3;d++)
		    {
		      // contribution to cell centroid
		      cx[d]+=((vscal*nn[d]/12.0)
			      *(xf[3*v0+d]*xf[3*v0+d]+
				xf[3*p1+d]*xf[3*p1+d]+
				xf[3*p2+d]*xf[3*p2+d]+
				xf[3*v0+d]*xf[3*p1+d]+
				xf[3*v0+d]*xf[3*p2+d]+
				xf[3*p1+d]*xf[3*p2+d]));
		      // contribution to face centroid
		      // by a given sub triangle
		      fcx[d]+=darea*(xf[3*v0+d]+xf[3*p1+d]+xf[3*p2+d])/3.0;
		    }
		  area+=darea;
		}	      
	    }
	  //
	  face_centroid[(3*f+0)*ncells+idx]=fcx[0]/area;
	  face_centroid[(3*f+1)*ncells+idx]=fcx[1]/area;
	  face_centroid[(3*f+2)*ncells+idx]=fcx[2]/area;
	  //printf("fc:%f %f %f\n",face_centroid[(3*f+0)*ncells+idx],face_centroid[(3*f+1)*ncells+idx],face_centroid[(3*f+2)*ncells+idx]);	  
	  //
	  normals[(3*f+0)*ncells+idx]=(norm[0]*vscal);
	  normals[(3*f+1)*ncells+idx]=(norm[1]*vscal);
	  normals[(3*f+2)*ncells+idx]=(norm[2]*vscal);	  
	}
      //printf("%lf %lf\n",volume[idx],vol/3.0);
      volume[idx]=(vol/3.0);
      centroid[idx]=cx[0]/volume[idx];
      centroid[ncells+idx]=cx[1]/volume[idx];
      centroid[2*ncells+idx]=cx[2]/volume[idx];
      //printf("cc:%f %f %f\n",centroid[idx],centroid[ncells+idx],centroid[2*ncells+idx]);
      //exit(0);
#if !defined (FVSAND_HAS_GPU)
      totalvol+=volume[idx];
      for(int f=0;f<ncon[idx];f++)
	{
	  double norm[3];
	  norm[0]=normals[(3*f+0)*ncells+idx];
	  norm[1]=normals[(3*f+1)*ncells+idx];
	  norm[2]=normals[(3*f+2)*ncells+idx];

	  normsum[0]+=norm[0];
	  normsum[1]+=norm[1];
	  normsum[2]+=norm[2];
	}
#endif
    }
#if !defined(FVSAND_HAS_GPU)
  //printf("normalsum: %lf %lf %lf %lf\n",normsum[0],normsum[1],normsum[2],totalvol);
#endif
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
	    //fprintf(fp,"%lf %lf %lf %lf %lf %lf\n",norm[0],norm[1],norm[2],norm1[0],norm1[1],norm1[2]);
#if !defined (FVSAND_HAS_GPU)
	    double norm[3];
	    double norm1[3];
            for(int d=0;d<3;d++)
             {
              norm[d]=normals[(3*(f-nccft[idx])+d)*ncells+idx];
              norm1[d]=normals[(3*(f1-nccft[idxn])+d)*ncells+idxn];
             }
	    conscheck+=(fabs(norm[0]+norm1[0])+
			fabs(norm[1]+norm1[1])+
			fabs(norm[2]+norm1[2]));
#endif	    
	  }
	}
    }
#if !defined (FVSAND_HAS_GPU)
  //printf("conscheck=%f\n",conscheck);
#endif
  //fclose(fp);
}

FVSAND_GPU_GLOBAL
void weighted_least_squares(double *weights, double *centroid,
			    double *facecentroid,
			    int *cell2cell, int *nccft,
			    int scale,int stride, int ncells)
{
#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells) 
#else
    for(int idx=0;idx<ncells;idx++)
#endif
      {
	double p[3];
	double xn[18];
	p[0]=centroid[scale*idx];
	p[1]=centroid[scale*idx+stride];
	p[2]=centroid[scale*idx+2*stride];
	double wmax=-1e15;
	for(int f=nccft[idx];f<nccft[idx+1];f++)
	  {
	    int idxn=cell2cell[f];
	    // if neighbor exists use it's centroid
	    if (idxn >=0 ) {
	      double ww=0;
	      for(int d=0;d<3;d++)
		{
		  ww+=(centroid[d*stride+scale*idxn]-p[d])*
		    (centroid[d*stride+scale*idxn]-p[d]);
		  xn[3*(f-nccft[idx])+d]=centroid[d*stride+scale*idxn];
		}
	      wmax=lsq_max(ww,wmax);
	    }
	    else
	      {
		// else use the centroid of the corresponding face
		double ww=0;
		for(int d=0;d<3;d++)
		  {
		    xn[3*(f-nccft[idx])+d]=facecentroid[(3*(f-nccft[idx])+d)*stride+idx];
		    ww+=(facecentroid[(3*(f-nccft[idx])+d)*stride+idx]-p[d])*
		      (facecentroid[(3*(f-nccft[idx])+d)*stride+idx]-p[d]);		    
		  }
		wmax=lsq_max(ww,wmax);
	      }
	  }
	int np=(nccft[idx+1]-nccft[idx]);
	double r11,r12,r22,r13,r23,r33;
	r11=r12=r22=r13=r23=r33=0;
	double dx[3];
	for(int f=0;f<np;f++)
	  {
	    double ww=0;
	    for(int d=0;d<3;d++) {
	      dx[d]=(xn[3*f+d]-p[d]);
	      ww+=(dx[d]*dx[d]);
	    }
	    ww=0.5*(1+cos(M_PI/KlsqCosFac*sqrt(ww/wmax)));
	    r11+=(ww*dx[0]*dx[0]);
	    r22+=(ww*dx[1]*dx[1]);
	    r33+=(ww*dx[2]*dx[2]);
	    r12+=(ww*dx[0]*dx[1]);
	    r13+=(ww*dx[0]*dx[2]);
	    r23+=(ww*dx[1]*dx[2]);	    
	  }
	r11=sqrt(lsq_max(KlsqMinVal,r11));
	r12=r12/r11;
	r22=lsq_max(KlsqMinVal,sqrt(r22-r12*r12));
	r13=r13/r11;
	r23=(r23-r12*r13)/r22;
	r33=sqrt(lsq_max(KlsqMinVal,r33-(r13*r13+r23*r23)));
	double b=(r12*r23-r13*r22)/(r11*r22);
	for(int f=0;f<np;f++)
	  {
	    double ww=0;
	    for(int d=0;d<3;d++) {
	      dx[d]=(xn[3*f+d]-p[d]);
	      ww+=(dx[d]*dx[d]);
	    }
	    ww=0.5*(1+cos(M_PI/KlsqCosFac*sqrt(ww/wmax)));
	    double a1=dx[0]/(r11*r11);
	    double a2=(dx[1]-r12*dx[0]/r11)/(r22*r22);
	    double a3=(dx[2]-r23*dx[1]/r22 + b*dx[0])/(r33*r33);
	    //
	    dx[0]=ww*(a1-a2*r12/r11+b*a3);
	    dx[1]=ww*(a2-a3*r23/r22);
	    dx[2]=ww*a3;
	    for(int d=0;d<3;d++)
	      weights[(3*f+d)*stride+idx*scale]=dx[d];
	  }	  
      }
}
