#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "parallelComm.h"
#include <cstdio>

using namespace fvSand;

LocalMesh::LocalMesh(GlobalMesh *g, int myid, MPI_Comm comm)
{
  parallelComm pc;
  
  // create communication patterns and ghost cells
  
  pc.createCommPatterns(myid,
			g->procmap,
			g->cell2cell,
			g->nconn,
			g->ncells,
			&ncells,
			&nhalo,
			cell2cell,
			ncon,
			local2global,
			global2local,
			sndmap,
			rcvmap,
			comm);
  //printf("ncells/nhalo=%d %d\n",ncells,nhalo);
  //
  // mine out the vertices of all the local cells
  // and turn everything into local coordinates
  //
  std::vector<int> iflag(g->nnodes,0);
  std::vector<uint64_t> cell2nodeg;
  
  //char fname[20];
  //sprintf(fname,"stats%d.dat",myid);
  //FILE *fp=fopen(fname,"w");

  nvcft.push_back(0);
  int m=0;
  //printf("local2global.size()=%d\n",local2global.size());
  //printf("ncells+nhalo=%d\n",ncells+nhalo);
  for(int i =0; i < ncells+nhalo;i++)
    {
      auto icell=local2global[i];
      int isum=0;
      int gptr=0;
      int n;
      //fprintf(fp,"%d %d ",icell,g->procmap[icell]);
      for(n=0;n<g->ntypes;n++)
        {
          isum+=g->nc[n];
          if (icell < isum)
            {
              gptr+=(icell-(isum-g->nc[n]))*g->nv[n];
              break;
            }
	  gptr+=(g->nc[n]*g->nv[n]);
	  
        }
      for(int j=0;j<g->nv[n];j++)
	{
	  cell2nodeg.push_back(g->cell2node[gptr+j]);
	  //fprintf(fp,"%d ",g->cell2node[gptr+j]);
	  //fflush(fp);
	  iflag[g->cell2node[gptr+j]]=1;
	}
      //fprintf(fp,"%d \n",g->nnodes);
      m++;
      nvcft.push_back(g->nv[n]+nvcft[m-1]);
    }
  //fclose(fp);
  nnodes=0;
  for(int i=0;i<g->nnodes;i++)
    if (iflag[i]) {
      x.push_back(g->x[3*i]);
      x.push_back(g->x[3*i+1]);
      x.push_back(g->x[3*i+2]);
      iflag[i]=nnodes;
      nnodes++;
    }
  for(auto c : cell2nodeg) cell2node.push_back(iflag[c]);

}

void LocalMesh::WriteMesh(int label)
{
  char fname[80];
  int i,j;
  FILE *fp;

  sprintf(fname,"localmesh%d.dat",label);
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"PMAP\"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEBLOCK\n",nnodes,
          ncells+nhalo);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED)\n");
  
  for(j=0;j<3;j++)
    for(i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d\n",1);
  for(i=0;i<nhalo;i++)
    fprintf(fp,"%d\n",-1);  
  
  for(i=0;i<ncells+nhalo;i++)
    {
      int m=0;
      int v[8];
      for(j=nvcft[i];j<nvcft[i+1];j++)
	v[m++]=(cell2node[j]+1);
      if (m==4) {
	fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		v[0],v[1],v[2],v[2],v[3],v[3],v[3],v[3]);
      }
      if (m==5) {
	fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		v[0],v[1],v[2],v[3],v[4],v[4],v[4],v[4]);
      }            
      if (m==6) {
	fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		v[0],v[1],v[2],v[2],v[3],v[4],v[5],v[5]);
      }
      if (m==8) {
	fprintf(fp,"%d %d %d %d %d %d %d %d\n",
		v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
      }
    }
  fclose(fp);
  
}
