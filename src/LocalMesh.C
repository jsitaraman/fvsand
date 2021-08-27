#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "parallelComm.h"
#include "fvsand_gpu.h"
#include "metric_functions.h"
#include "solver_functions.h"
#include <cstdio>

using namespace FVSAND;

LocalMesh::LocalMesh(GlobalMesh *g, int myid, MPI_Comm comm)
{
  mycomm=comm;
  int ierr=MPI_Comm_rank(comm,&myid);
  ierr=MPI_Comm_size(comm,&ngroup);
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
			mycomm);
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

void LocalMesh::createGridMetrics()
{
  x_d=gpu::push_to_device<double>(x.data(),sizeof(double)*x.size());
  cell2node_d=gpu::push_to_device<int>(cell2node.data(),sizeof(int)*cell2node.size());
  nvcft_d=gpu::push_to_device<int>(nvcft.data(),sizeof(int)*nvcft.size());
  ncon_d=gpu::push_to_device<int>(ncon.data(),sizeof(int)*ncon.size());
  
  center=gpu::allocate_on_device<double>(sizeof(double)*3*(ncells+nhalo));
  // allocate larger storage than necessary to avoid
  // unequal stride, 6=max faces for hex and 3 doubles per face
  normals=gpu::allocate_on_device<double>(sizeof(double)*18*(ncells+nhalo));

  // compute cell centers
  nthreads=(ncells+nhalo)*3;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_center,n_blocks,block_size,0,0,
			 center,x_d,nvcft_d,cell2node_d,ncells+nhalo);

  // compute cell normals
  
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_normals,n_blocks,block_size,0,0,
  			 normals,x_d,ncon_d,cell2node_d,nvcft_d,ncells+nhalo);
  
}

void LocalMesh::initSolution(double *flovar, int nfields)
{
 q=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 qn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 qnn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);

 nthreads=ncells+nhalo;
 //nthreads=ncells;
 n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
 int istor=0; // use row storage for now
 FVSAND_GPU_LAUNCH_FUNC(init_q,n_blocks,block_size,0,0,
		        q,center,flovar,nfields,istor,(ncells+nhalo));
 parallelComm pc;
 pc.exchangeDataDouble(q,nfields,(ncells+nhalo),istor,sndmap,rcvmap,mycomm);
 
}

  
void LocalMesh::WriteMesh(int label)
{
  char fname[80];
  int i,j;
  FILE *fp;

  sprintf(fname,"localmesh%d.dat",label);
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"PMAP\",\"Q0\",\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEBLOCK\n",nnodes,
          ncells+nhalo);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED, 5=CELLCENTERED)\n");
  
  for(j=0;j<3;j++)
    for(i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d\n",1);
  for(i=0;i<nhalo;i++)
    fprintf(fp,"%d\n",-1);
  for(i=0;i<ncells+nhalo;i++)
    fprintf(fp,"%lf\n",q[i*5+2]);
  
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
