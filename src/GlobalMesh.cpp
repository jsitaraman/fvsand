#include "GlobalMesh.h"
//
// read a surface mesh
// construct a simple strand mesh (no smoothing)
// find the cell to cell graph
//
extern "C" {
  void curl(double *, double *, double *);
  void normalize(double *);
}

class StrandMesh::StrandMesh(char *surface_file,double ds, double stretch, int nlevels)
{
  FILE *fp;  
  fp=fopen(surface_file,"r");  
  int nsurfnode,nsurfcells;
  fscanf(fp,"%d %d",&nsurfnode,&nsurfcells);
  std::vector<double> xsurf(3*nsurfnode);
  for(int i=0;i<nsurfnodes;i++)
    fscanf(fp,"%f %f %f",&(xsurf[3*i]),&(xsurf[3*i+1]),&(xsurf[3*i+2]));
  std::vector<int> tri(3*nsurfcells);
  for(int i=0;i<nsurfcells;i++)
    {
      fscanf(fp,"%d %d %d",&(tri[3*i]),&(tri[3*i+1]),&(tri[3*i+2]));
      tri[3*i]--;
      tri[3*i+1]--;
      tri[3*i+2]--;
    }
  fclose(fp);
  
  nnodes=nsurfnodes*nlevels;
  ncells=nsurfcells*(nlevels-1);
  
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
	  double V[3][3]
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
	  for(int j=0;j<3;j++)
	      x[3*m+j]=xsurf[3*i+j]+normals[3*i+j]*ds;
	  m++;
	}
      
      for(int i=0;i<nsurfcells;i++)
	{
	  for(int n=0;n<3;n++)
	    {
	      vconn[6*k+n]=tri[3*i+n]+offset;
	      vconn[6*k+n+3]=vconn[6*k+n]+nsurfnodes;	    
	    }
	  k++;
	}
      offset+=nsurfnodes;
      ds*=stretch;
    }  
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
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEBLOCK\n",blk[bid].nnodes,
          blk[bid].ncells);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED)\n");
  
  for(j=0;j<3;j++)
    for(i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d\n",procmap[i]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d %d %d %d %d %d %d %d\n",	    
	    conn[6*i]+1,
	    conn[6*i+1]+1,
	    conn[6*i+2]+1,
	    conn[6*i+2]+1,
	    conn[6*i+3]+1,
	    conn[6*i+4]+1,
	    conn[6*i+5]+1,
	    conn[6*i+5]+1);
  fclose(fp);
  
}
    
