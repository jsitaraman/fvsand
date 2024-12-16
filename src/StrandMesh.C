#include <cstdio>
#include <math.h>
#include "mpi.h"
#include "GlobalMesh.h"

#include "rcm.hpp"
#include "NVTXMacros.h"

using namespace FVSAND;
//
// read a surface mesh
// construct a simple strand mesh (no smoothing)
// find the cell to cell graph
//
extern "C" {
  void curl(double *, double *, double *);
  void normalize(double *);
  void get_exposed_faces_prizms_(int *,int *);
  void get_face_count_(int *,int *);
  void get_graph_(int *, int *, int *, int *);
  void getspherepart_(int *, int *, double *);
};

StrandMesh::StrandMesh(char* surface_file,double ds, double stretch, int nlevels, int myid)
{
  FILE *fp;  
  fp=fopen(surface_file,"r");
  if ( fp == nullptr ) {
    printf("Could not open file [%s]\n", surface_file );
    MPI_Abort( MPI_COMM_WORLD, -1 );
  }

  int nsurfnodes,nsurfcells;
  int ier;
  
  ier=fscanf(fp,"%d %d",&nsurfnodes,&nsurfcells);
  if (ier==0) {
   printf("File could not be read \n");
   exit(0);
  }
	
  std::vector<double> xsurf(3*nsurfnodes);
  for(int i=0;i<nsurfnodes && ier!=0;i++)
    ier=fscanf(fp,"%lf %lf %lf",&(xsurf[3*i]),&(xsurf[3*i+1]),&(xsurf[3*i+2]));
  if (ier==0) {
   printf("Coordinates could not be read \n");
   exit(0);
  }

  std::vector<int> tri(3*nsurfcells);
  for(int i=0;i<nsurfcells && ier!=0;i++)
    {
      ier=fscanf(fp,"%d %d %d",&(tri[3*i]),&(tri[3*i+1]),&(tri[3*i+2]));
      tri[3*i]--;
      tri[3*i+1]--;
      tri[3*i+2]--;
    }
  if (ier==0) {
   printf("Connectivity could not be read \n");
   exit(0);
  }
  fclose(fp);
  if (myid==0) printf("Finished reading grid ..\n");
  
  nnodes=nsurfnodes*(nlevels+1);
  ncells=nsurfcells*nlevels;
  
  int m=0;
  int k=0;
  int offset=0;
  std::vector<double> normals(3*nsurfnodes,0);
  //
  // create the storage for the mesh
  // use shared memory allocation with only
  // one rank per node doing all the memory
  // allocation and work
  //
  int nprocs,rank,nodesize,noderank;
  MPI_Win wintable1,wintable2,wintable3,wintable4,wintable5;
  MPI_Comm nodecomm;
  int windisp1,windisp2,windisp3,windisp4,windisp5;
  //
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
              MPI_INFO_NULL, &nodecomm);
  MPI_Comm_size(nodecomm,&nodesize);
  MPI_Comm_rank(nodecomm,&noderank);
  MPI_Aint xsize=0;
  MPI_Aint nsize=0;
  if (noderank==0) {
     xsize=3*nnodes;
     nsize=6*ncells;
   }
   windisp1=sizeof(double);
   MPI_Win_allocate_shared(xsize*sizeof(double), windisp1,
              MPI_INFO_NULL, nodecomm, &x, &wintable1);
   windisp2=sizeof(uint64_t);
   MPI_Win_allocate_shared(nsize*sizeof(uint64_t), windisp2,
              MPI_INFO_NULL, nodecomm, &cell2node, &wintable2);
   if (noderank!=0) {
        MPI_Win_shared_query(wintable1, 0, &xsize, &windisp1, &x);
        MPI_Win_shared_query(wintable2, 0, &nsize, &windisp2, &cell2node);
   }
   MPI_Win_fence(0,wintable1);
   MPI_Win_fence(0,wintable2);
	
  //x=new double [3*
  //	  nnodes];
  ntypes=1;
  nv=new int [ntypes];
  nv[0]=6;
  nc=new uint64_t [ntypes];
  nc[0]=ncells;
  procmap=new int [ncells];
  //cell2node=new uint64_t [6*ncells];
  if (noderank==0) { 
   for(int i=0;i<nsurfnodes;i++)
     {
      for(int j=0;j<3;j++)
	x[3*m+j]=xsurf[3*i+j];
      m++;
     }
		     
   for(int l=0;l<nlevels;l++)
    {
      for(int i=0;i<nsurfcells;i++)
	{
	  double V[3][3];
	  for (int n=0;n<3;n++)
	    for (int j=0;j<3;j++)
	      {
		V[n][j]=xsurf[3*tri[3*i+n]+j];
		if (n > 0) V[n][j]-=V[0][j];
	      }

	  double trinorm[3];
	  curl(V[1],V[2],trinorm);
	  normalize(trinorm);
	  for(int n=0;n<3;n++)
	    for(int j=0;j<3;j++)
	      normals[3*tri[3*i+n]+j]+=trinorm[j];
	}
      for(int i=0;i<nsurfnodes;i++)
	{
	  normalize(&(normals[3*i]));
	  for(int j=0;j<3;j++)
            {		  
	      x[3*m+j]=xsurf[3*i+j]+normals[3*i+j]*ds;
	      xsurf[3*i+j]=x[3*m+j];
	      normals[3*i+j]=0;
	    }
	  m++;
	}
      
      for(int i=0;i<nsurfcells;i++)
	{
	  for(int n=0;n<3;n++)
	    {
	      cell2node[6*k+n]=tri[3*i+n]+offset;
	      cell2node[6*k+n+3]=cell2node[6*k+n]+nsurfnodes;	    
	    }
	  k++;
	}
      offset+=nsurfnodes;
      ds*=stretch;
    }
  }
  if (myid==0) printf("Generated volume mesh ..\n");
  if (myid==0) printf("Total Prizmatic Elements: %ld\n",ncells);
  MPI_Win_fence(0,wintable1);
  MPI_Win_fence(0,wintable2);
  /* call canned f90 to get the neighbor information for all cells */
  MPI_Aint csize_m=0;
  MPI_Aint fsize_m=0;
  MPI_Aint ncells_m = 0;
  int *ctmp,*ftmp;
  int *ndc6,ntri,nquad;
  int csize,fsize;
  //
  if (noderank==0) {
   ndc6=new int[6*ncells];
   for(int i=0;i<6*ncells;i++) ndc6[i]=(int)cell2node[i];
   int ncells1=(int) ncells;	
  
   get_exposed_faces_prizms_(ndc6,&ncells1);
   get_face_count_(&ntri,&nquad);
   nfaces=ntri+nquad;
   csize=5*ncells;
   fsize=8*nfaces;
   ctmp=new int[csize];
   ftmp=new int[fsize];
   get_graph_(ctmp,ftmp,&csize,&fsize);
   ncells_m=ncells;
   csize_m=csize;
   fsize_m=fsize;
  }
  windisp3=sizeof(int64_t);
  MPI_Win_allocate_shared(csize_m*sizeof(int64_t), windisp3,
              MPI_INFO_NULL, nodecomm, &cell2cell, &wintable3);
  windisp4=sizeof(uint64_t);
  MPI_Win_allocate_shared(fsize_m*sizeof(uint64_t), windisp4,
              MPI_INFO_NULL, nodecomm, &faceInfo, &wintable4);
  windisp5=sizeof(int);
  MPI_Win_allocate_shared(ncells_m*sizeof(int), windisp5,
              MPI_INFO_NULL, nodecomm, &nconn, &wintable5);
  if (noderank!=0) {
     MPI_Win_shared_query(wintable3, 0, &csize_m, &windisp3, &cell2cell);
     MPI_Win_shared_query(wintable4, 0, &fsize_m, &windisp4, &faceInfo);
     MPI_Win_shared_query(wintable5, 0, &ncells_m, &windisp5, &nconn);
   }
  MPI_Win_fence(0,wintable3);
  MPI_Win_fence(0,wintable4);
  MPI_Win_fence(0,wintable5);
  //cell2cell = new int64_t[csize];
  //faceInfo  = new int64_t[fsize];
  //nconn     = new int[ncells];
  if (noderank==0) {
   for(int i=0;i<csize;i++) cell2cell[i]=(int64_t)(ctmp[i]);
   for(int i=0;i<fsize;i++) faceInfo[i]=(int64_t)(ftmp[i]);
   for(int i=0;i<ncells;i++) nconn[i]=5;
   k=0;
   int itype=2; // prizms
   for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      {
        if (cell2cell[5*i+j] < 0) {
	  int check_wall, check_outside;
	  check_wall=check_outside=1;
	  for(int v=0;v<numverts[itype][j];v++)
	    {
	      int f=face2node[itype][4*j+v]-1;
	      check_wall = (check_wall && (cell2node[6*i+f] < nsurfnodes));
	      check_outside=(check_outside && (cell2node[6*i+f] > nsurfnodes*nlevels-1));
	    }
	  if (check_wall) cell2cell[5*i+j]=-2;
	  if (check_outside) cell2cell[5*i+j]=-3;
  	 k++;
        }
      }
   WriteUgrid(0);
  }
  if (myid==0) printf("Assigned Boundary Conditions..\n");
  MPI_Win_fence(0,wintable3);
  MPI_Win_fence(0,wintable4);
  MPI_Win_fence(0,wintable5);
  //printf("k=%d\n",k);
  //WriteBoundaries(0);
  if (noderank==0) {
    delete [] ctmp;
    delete [] ftmp;
    delete [] ndc6;
  }
}

void StrandMesh::ReOrderCells(void) {
  int *adj_row = new int[ncells+1];
  adj_row[0]=1;
  for(int i=0;i<ncells;i++) {
    int nc = 0;
    for(int j=0;j<5;j++) {
      if (cell2cell[5*i+j] >= 0) {
        nc++;
      }
    }
    adj_row[i+1]=adj_row[i]+nc;  
  }

  int* adj = new int[adj_row[ncells]];

  int nc = 0;
  for(int i=0;i<ncells;i++) {
    for(int j=0;j<5;j++) {
      if (cell2cell[5*i+j] >= 0) {
        adj[nc] = (int)cell2cell[5*i+j]+1;
        nc++;
      }
    }
  }
  
  int *perm = new int[ncells];

  // Perform reverse Cuthill-McKee ordering
  genrcm(ncells,adj_row,adj,perm);

  delete [] adj_row;
  delete [] adj;

  for(int i=0;i<ncells;i++) {
    perm[i]--;
  }

  // Update cell2node
  uint64_t *cell2node_orig=new uint64_t [6*ncells];
  for(int i=0;i<6*ncells;i++) {
    cell2node_orig[i] = cell2node[i];
  }
  for(int i=0;i<ncells;i++) {
    nc = perm[i]; 
    for(int j=0;j<6;j++) {
      cell2node[6*nc+j] = cell2node_orig[6*i+j];
    }
  }
  delete [] cell2node_orig;

  // Update cell2cell
  int64_t *cell2cell_orig = new int64_t[5*ncells];
  for(int i=0;i<5*ncells;i++) {
    cell2cell_orig[i] = cell2cell[i];
  }

  for(int i=0;i<ncells;i++) {
    nc = perm[i]; 
    for(int j=0;j<5;j++) {
      if (cell2cell_orig[5*i+j] >= 0) {
         cell2cell[5*nc+j] = perm[cell2cell_orig[5*i+j]];
      }
      else {
         cell2cell[5*nc+j] = cell2cell_orig[5*i+j];
      }
    }
  }

  delete [] cell2cell_orig;

  delete [] perm;
  
}

void StrandMesh::WriteBoundaries(int label)
{

  int nsurfcells=0;
  for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      if (cell2cell[5*i+j] < 0) nsurfcells++;

  char fname[80];
  sprintf(fname,"strand_bc%d.dat",label);
  FILE *fp;

  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"DCF output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"PMAP\"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%ld E=%d ET=QUADRILATERAL, F=FEBLOCK\n",nnodes,
          nsurfcells);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED)\n");

  for(int j=0;j<3;j++)
    for(int i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      if (cell2cell[5*i+j] < 0) fprintf(fp,"%ld\n",cell2cell[5*i+j]);

  for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      if (cell2cell[5*i+j] < 0) {
	for(int k=0;k<4;k++)
	  fprintf(fp,"%ld ",cell2node[6*i+face2node[2][4*j+k]-1]+1);
	fprintf(fp,"\n");
      }
  fclose(fp);
}
void StrandMesh::PartitionSphereMesh(int myid,int numprocs,MPI_Comm comm)
{
  FVSAND_NVTX_FUNCTION( "partition" );

  double *arange=new double [4];
  int *pmap=new int [ncells];
  int mp1=myid+1;
  getspherepart_(&mp1,&numprocs,arange);
  int k=0;
  for(int i=0;i<ncells;i++)
    {
      pmap[i]=-1;
      double xc[3];
      xc[0]=xc[1]=xc[2]=0;
      for(int j=0;j<6;j++)
	for(int n=0;n<3;n++)
	  xc[n]+=x[3*(cell2node[6*i+j])+n];
      for(int n=0;n<3;n++)
	xc[n]*=0.16666666667;
      double theta=atan(xc[2]/sqrt(xc[1]*xc[1]+xc[0]*xc[0]));
      double phi=atan2(xc[1],xc[0]);
      if (phi < 0) phi+=(2*M_PI);
      if ((theta-arange[0])*(theta-arange[1]) <= 0.0 &&
	  (phi-arange[2])*(phi-arange[3]) <=0.0) {
	pmap[i]=myid;
	k++;
      }
    }
  int ierr=MPI_Allreduce(pmap,procmap,ncells,MPI_INT,MPI_MAX,comm);
}
void StrandMesh::WriteMesh(int label)
{
  char fname[80];
  int i,j;
  FILE *fp;

  sprintf(fname,"strandmesh%d.dat",label);
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"DCF output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"PMAP\"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%ld E=%ld ET=BRICK, F=FEBLOCK\n",nnodes,
          ncells);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED)\n");
  
  for(j=0;j<3;j++)
    for(i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d\n",procmap[i]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%ld %ld %ld %ld %ld %ld %ld %ld\n",	    
	    cell2node[6*i]+1,
	    cell2node[6*i+1]+1,
	    cell2node[6*i+2]+1,
	    cell2node[6*i+2]+1,
	    cell2node[6*i+3]+1,
	    cell2node[6*i+4]+1,
	    cell2node[6*i+5]+1,
	    cell2node[6*i+5]+1);
  fclose(fp);
  
}

void mwrite(int input,FILE *fp)
{
 union temp {
  int value;
  char c[4];
 } in,out;
 in.value=input;
 out.c[0]=in.c[3];
 out.c[1]=in.c[2];
 out.c[2]=in.c[1];
 out.c[3]=in.c[0];
 fwrite(&out.value,sizeof(int),1,fp); 	
}

void dwrite(double input,FILE *fp)
{
  union temp {
    double value;
    char c[8];
  } in,out;

  in.value = input;
  out.c[0] = in.c[7];
  out.c[1] = in.c[6];
  out.c[2] = in.c[5];
  out.c[3] = in.c[4];
  out.c[4] = in.c[3];
  out.c[5] = in.c[2];
  out.c[6] = in.c[1];
  out.c[7] = in.c[0];
  fwrite(&out.value,sizeof(out.value),1,fp);
}  

void StrandMesh::WriteUgrid(int label)
{
  char fname[80];
  int nsurfcells=0;
  for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      if (cell2cell[5*i+j] < 0) nsurfcells++;

  sprintf(fname,"strandmesh%d.b8.ugrid",label);
  FILE *fp=fopen(fname,"wb");
  mwrite(nnodes,fp);
  mwrite(nsurfcells,fp);
  mwrite(0,fp);
  mwrite(0,fp);
  mwrite(0,fp);
  mwrite(ncells,fp);
  mwrite(0,fp);
  for(int i=0;i<nnodes;i++) {    
    for(int j=0;j<3;j++)
      dwrite(x[3*i+j],fp);
  }
  for(int i=0;i<ncells;i++)
    for(int j=0;j<5;j++)
      if (cell2cell[5*i+j] < 0) {
	for(int k=0;k<3;k++) {
	  int indx=cell2node[6*i+face2node[2][4*j+k]-1]+1;
	  mwrite(indx,fp);
	 }
      }  
  for(int i=0;i<nsurfcells/2;i++) {
    int indx=1;
    mwrite(indx,fp);
  }
  for(int i=0;i<nsurfcells/2;i++)
  {
    int indx=2;
    mwrite(indx,fp);
  }
  for(int i=0;i<ncells;i++) {
    int indx[6];
    indx[0]=cell2node[6*i+4]+1;
    indx[1]=cell2node[6*i+3]+1;
    indx[2]=cell2node[6*i+5]+1;
    indx[3]=cell2node[6*i+1]+1;
    indx[4]=cell2node[6*i+0]+1;
    indx[5]=cell2node[6*i+2]+1;
    for(int k=0;k<6;k++)
      mwrite(indx[k],fp);
  }
  fclose(fp);
}
