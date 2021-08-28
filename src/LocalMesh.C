#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
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

  for(auto s: sndmap) 
    for (auto v : s.second)
      device2host.push_back(v);
  
  for(auto r: rcvmap) 
    for (auto v : r.second)
      host2device.push_back(v);

  device2host_d=gpu::push_to_device<int>(device2host.data(),sizeof(int)*device2host.size());
  host2device_d=gpu::push_to_device<int>(host2device.data(),sizeof(int)*host2device.size());

}

void LocalMesh::CreateGridMetrics()
{
  x_d=gpu::push_to_device<double>(x.data(),sizeof(double)*x.size());
  cell2node_d=gpu::push_to_device<int>(cell2node.data(),sizeof(int)*cell2node.size());  
  nvcft_d=gpu::push_to_device<int>(nvcft.data(),sizeof(int)*nvcft.size());
  ncon_d=gpu::push_to_device<int>(ncon.data(),sizeof(int)*ncon.size());
  cell2cell_d=gpu::push_to_device<int>(cell2cell.data(),sizeof(int)*cell2cell.size());

  // create cell frequency table
  nccft_h=new int[ncon.size()+1];
  nccft_h[0]=0;
  for(int i=0;i<ncells+nhalo;i++)
    nccft_h[i+1]=nccft_h[i]+ncon[i];  
  nccft_d=gpu::push_to_device<int>(nccft_h,sizeof(int)*(ncon.size()+1));  

  // allocate storage for metrics
  center=gpu::allocate_on_device<double>(sizeof(double)*3*(ncells+nhalo));
  // allocate larger storage than necessary for normals to avoid
  // unequal stride, 6=max faces for hex and 3 doubles per face
  normals=gpu::allocate_on_device<double>(sizeof(double)*18*(ncells+nhalo));
  volume=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo));
  
  // compute cell centers
  nthreads=(ncells+nhalo)*3;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_center,n_blocks,block_size,0,0,
			 center,x_d,nvcft_d,cell2node_d,ncells+nhalo);

  // compute cell normals and volume
  
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_normals_volume,n_blocks,block_size,0,0,
  			 normals,volume,x_d,ncon_d,cell2node_d,nvcft_d,ncells+nhalo);

  // check conservation
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(check_conservation,n_blocks,block_size,0,0,
			 normals,x_d,nccft_d,cell2cell_d,ncells+nhalo);
}

void LocalMesh::InitSolution(double *flovar, int nfields)
{
 q=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 qn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 qnn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 res=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 
 flovar_d=gpu::push_to_device<double>(flovar,sizeof(double)*nfields);
 qinf_d=gpu::allocate_on_device<double>(sizeof(double)*nfields);
 
 nfields_d=nfields;
 nthreads=ncells+nhalo;
 //nthreads=ncells;
 n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
 FVSAND_GPU_LAUNCH_FUNC(init_q,n_blocks,block_size,0,0,
		        qinf_d,q,center,flovar_d,nfields,istor,(ncells+nhalo));

 qh=new double [sizeof(double)*(ncells+nhalo)*nfields];
 gpu::pull_from_device<double>(qh,q,sizeof(double)*(ncells+nhalo)*nfields);

 int dsize=device2host.size();
 dsize=(dsize < host2device.size())?host2device.size():dsize;
 qbuf=new double [dsize*nfields];
 qbuf_d=gpu::allocate_on_device<double>(sizeof(double)*dsize);

}

void LocalMesh::UpdateFringes(double *qh, double *qd)
{
  nthreads=device2host.size();
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateHost,n_blocks,block_size,0,0,
			 qbuf_d,q,device2host_d,device2host.size());
  gpu::pull_from_device<double>(qbuf,qbuf_d,sizeof(double)*device2host.size());
  for(int i=0;i<device2host.size();i++)
    qh[device2host[i]]=qbuf[i];
  
  pc.exchangeDataDouble(qh,nfields_d,(ncells+nhalo),istor,sndmap,rcvmap,mycomm);

  for(int i=0;i<host2device.size();i++)
    qbuf[i]=qh[host2device[i]];

  gpu::copy_to_device(qbuf_d,qbuf,sizeof(double)*host2device.size());

  nthreads=host2device.size();
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateDevice,n_blocks,block_size,0,0,
			 q,qbuf_d,host2device_d,host2device.size());
}
				
				
void LocalMesh::Residual(double *qv)
{
  nthreads=ncells;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(computeResidual,n_blocks,block_size,0,0,
			 res, qv, center, normals, volume,
			 qinf_d, cell2cell_d, nccft_d, nfields_d,istor,ncells);
  parallelComm pc;
  pc.exchangeDataDouble(res,nfields_d,(ncells+nhalo),istor,sndmap,rcvmap,mycomm);

}

void LocalMesh::Update(double *qdest, double *qsrc, double fscal)
{
  nthreads=(ncells+nhalo)*nfields_d;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateFields,n_blocks,block_size,0,0,
			 res, qdest, qsrc, fscal, (ncells+nhalo)*nfields_d);
}

double LocalMesh::ResNorm()
{
  gpu::pull_from_device<double>(qh,res,sizeof(double)*nfields_d*(ncells+nhalo));
  double rnorm[2];
  double rnormTotal[2];

  rnorm[0]=0.0;
  rnorm[1]=ncells;
  for(int i=0;i<nfields_d*ncells;i++)
    rnorm[0]+=(qh[i]*qh[i]);
  MPI_Reduce(rnorm,rnormTotal,2,MPI_DOUBLE,MPI_SUM, 0, mycomm);
  
  rnormTotal[0]=sqrt(rnormTotal[0]/nfields_d/rnormTotal[1]);
  return rnormTotal[0];
}
  
void LocalMesh::WriteMesh(int label)
{
  char fname[80];
  int i,j,n;
  FILE *fp;

  sprintf(fname,"localmesh%d.dat",label);
  fp=fopen(fname,"w");
  fprintf(fp,"TITLE =\"output\"\n");
  fprintf(fp,"VARIABLES=\"X\",\"Y\",\"Z\",\"PMAP\",\"VOL\",");
  for(n=0;n<nfields_d;n++)
    fprintf(fp,"\"Q%d\",",n);
  fprintf(fp,"\n");
  fprintf(fp,"ZONE T=\"VOL_MIXED\",N=%d E=%d ET=BRICK, F=FEBLOCK\n",nnodes,
          ncells+nhalo);
  fprintf(fp,"VARLOCATION = (1=NODAL, 2=NODAL, 3=NODAL, 4=CELLCENTERED, 5=CELLCENTERED, ");
  for(n=0;n<nfields_d;n++)
    fprintf(fp,"%d=CELLCENTERED, ",n+6);
  fprintf(fp,")\n");
  
  for(j=0;j<3;j++)
    for(i=0;i<nnodes;i++) fprintf(fp,"%.14e\n",x[3*i+j]);
  for(i=0;i<ncells;i++)
    fprintf(fp,"%d\n",1);
  for(i=0;i<nhalo;i++)
    fprintf(fp,"%d\n",-1);

  gpu::pull_from_device<double>(qh,volume,sizeof(double)*(ncells+nhalo));
  for(i=0;i<ncells+nhalo;i++)
    fprintf(fp,"%lf\n",qh[i]);

  gpu::pull_from_device<double>(qh,res,sizeof(double)*nfields_d*(ncells+nhalo));
  for(n=0;n<nfields_d;n++)
    for(i=0;i<ncells+nhalo;i++)
      fprintf(fp,"%lf\n",qh[i*nfields_d+n]);
  
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
