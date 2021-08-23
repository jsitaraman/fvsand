#include <cstdio>
#include "GlobalMesh.h"

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
};

StrandMesh::StrandMesh(char* surface_file,double ds, double stretch, int nlevels)
{
  FILE *fp;  
  fp=fopen(surface_file,"r");  
  int nsurfnodes,nsurfcells;
  int ier;
  ier=fscanf(fp,"%d %d",&nsurfnodes,&nsurfcells);
  std::vector<double> xsurf(3*nsurfnodes);
  for(int i=0;i<nsurfnodes;i++)
    ier=fscanf(fp,"%lf %lf %lf",&(xsurf[3*i]),&(xsurf[3*i+1]),&(xsurf[3*i+2]));
  std::vector<int> tri(3*nsurfcells);
  for(int i=0;i<nsurfcells;i++)
    {
      ier=fscanf(fp,"%d %d %d",&(tri[3*i]),&(tri[3*i+1]),&(tri[3*i+2]));
      tri[3*i]--;
      tri[3*i+1]--;
      tri[3*i+2]--;
    }
  fclose(fp);
  
  nnodes=nsurfnodes*(nlevels+1);
  ncells=nsurfcells*nlevels;
  
  int m=0;
  int k=0;
  int offset=0;
  std::vector<double> normals(3*nsurfnodes,0);

  // create the storage for the mesh
  
  x=new double [3*nnodes];
  nv=new int [1];
  nv[0]=6;
  nc=new uint64_t [1];
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

  /* call canned f90 to get the neighbor information for all cells */
  
  int *ndc6,ntri,nquad;
  ndc6=new int[6*ncells];
  for(int i=0;i<6*ncells;i++) ndc6[i]=(int)cell2node[i];
  int ncells1=(int) ncells;	
  printf("ncells1=%d\n",ncells1);
  
  get_exposed_faces_prizms_(ndc6,&ncells1);
  get_face_count_(&ntri,&nquad);
  nfaces=ntri+nquad;
  printf("nfaces=%ld\n",nfaces);
  int *ctmp,*ftmp;
  int csize=5*ncells;
  int fsize=8*nfaces;
  ctmp=new int[csize];
  ftmp=new int[fsize];
  get_graph_(ctmp,ftmp,&csize,&fsize);
  cell2cell = new int64_t[csize];
  faceInfo  = new int64_t[fsize];
  
  for(int i=0;i<csize;i++) cell2cell[i]=(int64_t)(ctmp[i]);
  for(int i=0;i<fsize;i++) faceInfo[i]=(int64_t)(ftmp[i]);
  
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
  //printf("k=%d\n",k);
  //WriteBoundaries(0);

  delete [] ctmp;
  delete [] ftmp;
  delete [] ndc6;
  
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
    
