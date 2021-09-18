#include <cstdio>
#include <math.h>
#include "mpi.h"
#include "GlobalMesh.h"

#include "NVTXMacros.h"
#define NSP 20

extern "C" {
  extern void distrib_(int *, double *, double *, double *, double *);
}

using namespace FVSAND;
//
// read x-, y-, and z- distributions
// and create a Cartesian rectilinear mesh
//
CartesianMesh::CartesianMesh(char *cart_file)
{
  FILE *fp;
  int NN[3];
  char line[256];
  int   idiv[3];
  double arc[3][NSP],ds[3][NSP],idim[3][NSP];
  double *xc[3];
  
  fp=fopen(cart_file,"r");
  if ( fp == nullptr ) {
    printf("Could not open file [%s]\n", cart_file );
    MPI_Abort( MPI_COMM_WORLD, -1 );
  }
  while((fgetc(fp))!='\n');
  fscanf(fp,"%d %d %d",&NN[0],&NN[1],&NN[2]);
  printf("%d %d %d\n",NN[0],NN[1],NN[2]);
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
  for(int d=0;d<3;d++) {
   for(int j=0;j<idiv[d];j++) printf("%lf ",idim[d][j]);
   printf("\n");
   for(int j=0;j<idiv[d];j++) printf("%lf ",arc[d][j]);
   printf("\n");
   for(int j=0;j<idiv[d];j++) printf("%lf ",ds[d][j]);
   printf("\n--------------------------\n");
  }

  for(int d=0;d<3;d++)
    {
      xc[d]=(double *)malloc(sizeof(double)*NN[d]);
      distrib_(&idiv[d],idim[d],arc[d],ds[d],xc[d]);
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
  // form the connectivity of the rectilinear mesh
  m=0;
  for(int k=0;k<NN[2]-1;k++)
    for(int j=0;j<NN[1]-1;j++)
      for(int i=0;i<NN[0]-1;i++)
	{
	  procmap[m]=0;
	  cell2node[8*m]=(k-1)*NN[1]*NN[0]+(j-1)*NN[0]+i;
	  cell2node[8*m+1]=cell2node[8*m]+1;
	  cell2node[8*m+2]=cell2node[8*m+1]+NN[0];
	  cell2node[8*m+3]=cell2node[8*m]+NN[0];
	  for(int p=4;p<8;p++)
	    cell2node[8*m+p]=cell2node[8*m+p-4]+NN[1]*NN[0];
	  m++;
	}
  
  delete [] xc[0];
  delete [] xc[1];
  delete [] xc[2];

  
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

