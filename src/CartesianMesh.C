#include <cstdio>
#include <math.h>
#include "mpi.h"
#include "GlobalMesh.h"

#include "NVTXMacros.h"
#define NSP 20

extern "C"
{
  extern void distrib_(int *, double *, double *, double *, double *);
  extern void factorize_(int *, int *);
  extern void divide1d_(int *, int *, int *);
  void get_exposed_faces_hexs_(int *,int *);
  void get_face_count_(int *,int *);
  void get_graph_(int *, int *, int *, int *);
}

using namespace FVSAND;
//
// read x-, y-, and z- distributions
// and create a Cartesian rectilinear mesh
//
CartesianMesh::CartesianMesh(char *cart_file,int numprocs)
{
  FILE *fp;
  int NN[3];
  char line[256];
  int   idiv[3],nx[3],*displ[3];
  double arc[3][NSP],ds[3][NSP],idim[3][NSP];
  double *xc[3];
  
  fp=fopen(cart_file,"r");
  if ( fp == nullptr ) {
    printf("Could not open file [%s]\n", cart_file );
    MPI_Abort( MPI_COMM_WORLD, -1 );
  }
  while((fgetc(fp))!='\n');
  fscanf(fp,"%d %d %d",&NN[0],&NN[1],&NN[2]);
  //printf("%d %d %d\n",NN[0],NN[1],NN[2]);
  while((fgetc(fp))!='\n');
  for(int d=0;d<3;d++)
    {
      while((fgetc(fp))!='\n');
      fgets(line,256,fp);
      double f;
      int total_n,n,j;
      total_n=0;
      int m=0;
      while(1==sscanf(line + total_n, "%d%n", &j, &n))
	{
	  total_n+=n;
	  idim[d][m++]=j;
	}
      fgets(line,256,fp);
      idiv[d]=m;
      m=0;
      total_n=0;
      while(1==sscanf(line + total_n, "%lf%n", &f, &n) && m < idiv[d])
	{
	  total_n+=n;
	  arc[d][m++]=f;
	}
      m=0;
      fgets(line,256,fp);
      total_n=0;
      while(1==sscanf(line + total_n, "%lf%n", &f, &n) && m < idiv[d])
	{
	  total_n+=n;
	  ds[d][m++]=f;
	}
    }
  /*
  for(int d=0;d<3;d++) {
   for(int j=0;j<idiv[d];j++) printf("%lf ",idim[d][j]);
   printf("\n");
   for(int j=0;j<idiv[d];j++) printf("%lf ",arc[d][j]);
   printf("\n");
   for(int j=0;j<idiv[d];j++) printf("%lf ",ds[d][j]);
   printf("\n--------------------------\n");
  } */

  for(int d=0;d<3;d++)
    {
      xc[d]=(double *)malloc(sizeof(double)*NN[d]);
      if (idiv[d]==2) {
        for(int i=0;i<NN[d];i++)
	  xc[d][i]=(arc[d][0]+i*(arc[d][1]-arc[d][0])/(NN[d]-1));
      } else {
      distrib_(&idiv[d],idim[d],arc[d],ds[d],xc[d]);
      }
    }
  for(int i=0;i<NN[0];i++)
    printf("%lf %lf %lf\n",xc[0][i],xc[1][i],xc[2][i]);

  // create a hexahedral unstructured mesh
  nnodes=NN[0]*NN[1]*NN[2]; 
  x=new double [3*nnodes];
  ncells=(NN[0]-1)*(NN[1]-1)*(NN[2]-1);
  ntypes=1;
  nv=new int [ntypes];
  nv[0]=8;
  nc= new uint64_t[ntypes];
  nc[0]=ncells;
  procmap= new int [ncells];
  cell2node = new uint64_t[nv[0]*ncells];
  // form the coordinates of the rectilinear mesh
  int m=0;
  for(int k=0;k<NN[2];k++)
    for(int j=0;j<NN[1];j++)
      for(int i=0;i<NN[0];i++)
	{
	  x[m++]=xc[0][i];
	  x[m++]=xc[1][j];
	  x[m++]=xc[2][k];
	}
  //
  // factorize the number of processors into
  // three integers and create divisions
  // of the nodes in each of the three directions
  //
  factorize_(&numprocs,nx);
  //printf("%d %d %d\n",nx[0],nx[1],nx[2]);
  for(int d=0;d<3;d++)
    {
      displ[d]=new int[nx[d]+1];
      displ[d][0]=0;
      divide1d_(&nx[d],&NN[d],&displ[d][1]);
      //for(int k=0;k<nx[d]+1;k++) printf("%d ",displ[d][k]);
      //printf("\n");
    }
  // form the connectivity of the rectilinear mesh
  // and assign proc maps based on the division created
  // above
  m=0;
  int ix[3];
  int px[3];
  //int *cellcount= new int [numprocs]{0};
  //
  px[2]=0;
  ix[2]=1;
  for(int k=0;k<NN[2]-1;k++)
    {
      if ( k+0.5 > displ[2][ix[2]]) {
	px[2]++;
	ix[2]++;
      }
      px[1]=0;
      ix[1]=1;
      for(int j=0;j<NN[1]-1;j++)
	{
	  if ( j+0.5 > displ[1][ix[1]]) {
	    px[1]++;
	    ix[1]++;
	  }
	  px[0]=0;
	  ix[0]=1;
	  for(int i=0;i<NN[0]-1;i++)
	    {
	      if (i+0.5 > displ[0][ix[0]]) {
		px[0]++;
		ix[0]++;
	      }
	      procmap[m]=px[2]*nx[1]*nx[0]+px[1]*nx[0]+px[0];
	      //cellcount[procmap[m]]++;
	      cell2node[8*m]=static_cast<uint64_t> (k*NN[1]*NN[0]+j*NN[0]+i);
	      cell2node[8*m+1]=static_cast<uint64_t>(cell2node[8*m]+1);
	      cell2node[8*m+2]=static_cast<uint64_t>(cell2node[8*m+1]+NN[0]);
	      cell2node[8*m+3]=static_cast<uint64_t>(cell2node[8*m]+NN[0]);
	      for(int p=4;p<8;p++)
		cell2node[8*m+p]=static_cast<uint64_t>(cell2node[8*m+p-4]+NN[1]*NN[0]);
	      //for(int p=0;p<8;p++)
	      //  if (cell2node[8*m+p] < 0) printf("i,j,k,p=%d %d %d %d\n",i,j,k,p);
	      m++;
	    }
	}
    }

  int *ndc8,ntri,nquad;
  ndc8= new int [8*ncells];
  for(int i=0;i<8*ncells;i++) ndc8[i]=(int)cell2node[i];
  int ncells1=(int) ncells;
  get_exposed_faces_hexs_(ndc8,&ncells1);
  get_face_count_(&ntri,&nquad);
  nfaces=ntri+nquad;
  int *ctmp,*ftmp;
  int csize=6*ncells;
  int fsize=8*nfaces;
  ctmp=new int[csize];
  ftmp=new int[fsize];
  get_graph_(ctmp,ftmp,&csize,&fsize);
  cell2cell = new int64_t[csize];
  faceInfo  = new int64_t[fsize];
  nconn     = new int[ncells];
  
  for(int i=0;i<csize;i++) cell2cell[i]=(int64_t)(ctmp[i]);
  for(int i=0;i<fsize;i++) faceInfo[i]=(int64_t)(ftmp[i]);
  for(int i=0;i<ncells;i++) nconn[i]=6;
  int k=0;
  int itype=3;
  for(int i=0;i<ncells;i++)
    for(int j=0;j<nconn[i];j++)
      {
        if (cell2cell[nconn[i]*i+j] < 0) {
	  cell2cell[nconn[i]*i+j]=-3;
  	 k++;
        }
      }
  printf("k=%d\n",k);
  WriteBoundaries(0);
  
  //for(int i=0;i<numprocs;i++) printf("%d\n",cellcount[i]);

  delete [] ctmp;
  delete [] ftmp;
  delete [] ndc8;
  
  for(int d=0;d<3;d++)
    {
     delete [] xc[d];
     delete [] displ[d];
    }
  
}

void CartesianMesh::WriteMesh(int label)
{
  char fname[80];
  int i,j;
  FILE *fp;

  sprintf(fname,"CartMesh%d.dat",label);
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
	    cell2node[8*i]+1,
	    cell2node[8*i+1]+1,
	    cell2node[8*i+2]+1,
	    cell2node[8*i+3]+1,
	    cell2node[8*i+4]+1,
	    cell2node[8*i+5]+1,
	    cell2node[8*i+6]+1,
	    cell2node[8*i+7]+1);
  fclose(fp);
  
}

void CartesianMesh::WriteBoundaries(int label)
{

  int nsurfcells=0;
  for(int i=0;i<ncells;i++)
    for(int j=0;j<6;j++)
      if (cell2cell[6*i+j] < 0) nsurfcells++;

  char fname[80];
  sprintf(fname,"cart_bc%d.dat",label);
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
    for(int j=0;j<6;j++)
      if (cell2cell[6*i+j] < 0) fprintf(fp,"%ld\n",cell2cell[6*i+j]);

  for(int i=0;i<ncells;i++)
    for(int j=0;j<6;j++)
      if (cell2cell[6*i+j] < 0) {
	for(int k=0;k<4;k++)
	  fprintf(fp,"%ld ",cell2node[8*i+face2node[3][4*j+k]-1]+1);
	fprintf(fp,"\n");
      }
  fclose(fp);
}

void CartesianMesh::WriteUgrid(int label)
{
  char fname[80];
  int k2[3]={0,2,3};
  
  int nsurfcells=0;
  for(int i=0;i<ncells;i++)
    for(int j=0;j<6;j++)
      if (cell2cell[6*i+j] < 0) nsurfcells++;

  sprintf(fname,"cartmesh%d.ugrid",label);
  FILE *fp=fopen(fname,"w");
  fprintf(fp,"%ld %d %d %d %d %d %d %ld\n",nnodes,0,nsurfcells,0,0,0,0,ncells);
  for(int i=0;i<nnodes;i++) {    
    for(int j=0;j<3;j++)
      fprintf(fp,"%.14e ",x[3*i+j]);
    fprintf(fp,"\n");
  }
  for(int i=0;i<ncells;i++)
    for(int j=0;j<6;j++)
      if (cell2cell[6*i+j] < 0) {
	{
	  for(int k=0;k<4;k++)
	    fprintf(fp,"%ld ",cell2node[8*i+face2node[3][4*j+k]-1]+1);
	  fprintf(fp,"\n");
	}
      }
  for(int i=0;i<nsurfcells;i++) {
      fprintf(fp,"%d\n",1);
   }

  for(int i=0;i<ncells;i++) {
    fprintf(fp,"%ld %ld %ld %ld %ld %ld %ld %ld\n",	    
	    cell2node[8*i+0]+1,
	    cell2node[8*i+1]+1,
	    cell2node[8*i+2]+1,
	    cell2node[8*i+3]+1,
	    cell2node[8*i+4]+1,
	    cell2node[8*i+5]+1,
	    cell2node[8*i+6]+1,
	    cell2node[8*i+7]+1);
  }
  fclose(fp);
}
