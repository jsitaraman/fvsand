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
void curlp(double *A, double *B , double *result)
{
  result[0]+=(A[1]*B[2]-B[1]*A[2]);
  result[1]+=(A[2]*B[0]-B[2]*A[0]);
  result[2]+=(A[0]*B[1]-B[0]*A[1]);
}

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
void cell_normals(double *normals,
		  double *x,
		  int *ncon,
		  int *cell2node,
		  int *nvcft,
		  int ncells)
{

#if defined (FVSAND_HAS_GPU)
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < ncells*3) 
#else
  double normsum[3];
  normsum[0]=normsum[1]=normsum[2]=0;
  for(int idx=0;idx<ncells;idx++)
#endif
    {
      int m=0;
      int v[8];
      for(int j=nvcft[idx];j<nvcft[idx+1];j++)
	v[m++]=cell2node[j];

      // kludge to convert [4,5,6,8] to [0,1,2,3]
      int itype=(nvcft[idx+1]-nvcft[idx]-4);
      itype=(itype > 3) ? 3: itype;
      
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
	  
	  for(int p=0;p<numverts[itype][f]-2;p++)
	    {
	      A[0]=xf[3*(p+1)  ]-xf[0];
	      A[1]=xf[3*(p+1)+1]-xf[1];
	      A[2]=xf[3*(p+1)+2]-xf[2];
	      
	      B[0]=xf[3*(p+2)  ]-xf[0];
	      B[1]=xf[3*(p+2)+1]-xf[1];
	      B[2]=xf[3*(p+2)+2]-xf[2];	      
	      // remember the face connectivity is 
	      // facing towards the interior of the cell
	      curlp(B,A,norm);
	    }
	}
      for(int f=0;f<ncon[idx];f++)
	{
	  double *norm=normals+18*idx+3*f;
	  normsum[0]+=norm[0];
	  normsum[1]+=norm[1];
	  normsum[2]+=norm[2];
	}
    }
    printf("normalsum: %lf %lf %lf\n",normsum[0],normsum[1],normsum[2]);
}    
