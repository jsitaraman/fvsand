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

  // create the storage for the mesh
  
  x=new double [3*nnodes];
  ntypes=1;
  nv=new int [ntypes];
  nv[0]=6;
  nc=new uint64_t [ntypes];
  nc[0]=ncells;
  procmap=new int [ncells];
  cell2node=new uint64_t [6*ncells];
  
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
  if (myid==0) printf("Generated volume mesh ..\n");
  if (myid==0) printf("Total Prizmatic Elements: %d\n",ncells);

  /* call canned f90 to get the neighbor information for all cells */
  
  int *ndc6,ntri,nquad;
  ndc6=new int[6*ncells];
  for(int i=0;i<6*ncells;i++) ndc6[i]=(int)cell2node[i];
  int ncells1=(int) ncells;	
  
  get_exposed_faces_prizms_(ndc6,&ncells1);
  get_face_count_(&ntri,&nquad);
  nfaces=ntri+nquad;
  int *ctmp,*ftmp;
  int csize=5*ncells;
  int fsize=8*nfaces;
  ctmp=new int[csize];
  ftmp=new int[fsize];
  get_graph_(ctmp,ftmp,&csize,&fsize);
  cell2cell = new int64_t[csize];
  faceInfo  = new int64_t[fsize];
  nconn     = new int[ncells];

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
  if (myid==0) printf("Assigned Boundary Conditions..\n");
  //printf("k=%d\n",k);
  //WriteBoundaries(0);

  delete [] ctmp;
  delete [] ftmp;
  delete [] ndc6;

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
    
