#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "fvsand_gpu.h"
#include "metric_functions.h"
#include "solver_functions.h"
#include "NVTXMacros.h"
#include <cstdio>

using namespace FVSAND;

LocalMesh::~LocalMesh()
{
  // release memory in the order they are declared
  // in LocalMesh.h
  FVSAND_FREE_DEVICE(device2host_d);
  FVSAND_FREE_DEVICE(host2device_d);
  //
  if (qbuf) delete [] qbuf;
  if (qbuf2) delete [] qbuf2;  
  FVSAND_FREE_DEVICE(qbuf_d);
  FVSAND_FREE_DEVICE(qbuf_d2);
  //
  FVSAND_FREE_DEVICE(x_d);
  FVSAND_FREE_DEVICE(flovar_d);
  FVSAND_FREE_DEVICE(qinf_d);
  //
  FVSAND_FREE_DEVICE(cell2node_d);
  FVSAND_FREE_DEVICE(nvcft_d);
  if (nccft_h) delete [] nccft_h;
  FVSAND_FREE_DEVICE(nccft_d);
  FVSAND_FREE_DEVICE(ncon_d);
  //
  FVSAND_FREE_DEVICE(cell2cell_d);
  FVSAND_FREE_DEVICE(center_d);
  FVSAND_FREE_DEVICE(normals_d);
  FVSAND_FREE_DEVICE(volume_d);
  FVSAND_FREE_DEVICE(res_d);
  //
  FVSAND_FREE_DEVICE(rmatall_d);
  FVSAND_FREE_DEVICE(Dall_d);
  //
  FVSAND_FREE_DEVICE(cell2face_d);
  FVSAND_FREE_DEVICE(facetype_d);
  FVSAND_FREE_DEVICE(facenorm_d);
  FVSAND_FREE_DEVICE(faceq_d);
  FVSAND_FREE_DEVICE(faceflux_d);
  FVSAND_FREE_DEVICE(lsqwts);
  //
  if (ireq) delete [] ireq;
  if (istatus) delete [] istatus;
  //
  if (qh) delete [] qh;
  FVSAND_FREE_DEVICE(q);
  FVSAND_FREE_DEVICE(qn);
  FVSAND_FREE_DEVICE(qnn);
  FVSAND_FREE_DEVICE(dq_d);
  FVSAND_FREE_DEVICE(dqupdate_d);  
}

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
  //
  // build local cells and local mesh connectivity
  //
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

void LocalMesh::CreateGridMetrics()
{
  FVSAND_NVTX_FUNCTION( "grid_metrics" );

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

  Dall_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*25);
  rmatall_d=gpu::allocate_on_device<double>(sizeof(double)*nccft_h[ncells+nhalo]*25);

  // allocate storage for metrics
  center_d=gpu::allocate_on_device<double>(sizeof(double)*3*(ncells+nhalo));
  // allocate larger storage than necessary for normals to avoid
  // unequal stride, 6=max faces for hex and 3 doubles per face
  normals_d=gpu::allocate_on_device<double>(sizeof(double)*18*(ncells+nhalo));
  volume_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo));
  
  // compute cell center_ds
  nthreads=(ncells+nhalo)*3;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_center,n_blocks,block_size,0,0,
			 center_d,x_d,nvcft_d,cell2node_d,ncells+nhalo);

  // compute cell normals and volume_d
  
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(cell_normals_volume,n_blocks,block_size,0,0,
  			 normals_d,volume_d,x_d,ncon_d,cell2node_d,nvcft_d,ncells+nhalo);

  // check conservation
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(check_conservation,n_blocks,block_size,0,0,
			 normals_d,x_d,nccft_d,cell2cell_d,ncells+nhalo);

  CreateFaces();
}

void LocalMesh::CreateFaces(void)
{
  // find cell2face connectivity
  // for face based residual calculation
  
  std::vector<double> facenorm_h;
  std::vector<int> facetype_h;
  std::vector<int>iflag(cell2cell.size()+1,1);
  int *cell2face_h=new int [cell2cell.size()];
  double *normals_h = new double [(ncells+nhalo)*18];
  gpu::pull_from_device<double>(normals_h,normals_d,sizeof(double)*(ncells+nhalo)*18);

  nfaces=0;
  for(int idx=0;idx<(ncells+nhalo);idx++)
    for(int f=nccft_h[idx];f<nccft_h[idx+1];f++)
      {
	if (iflag[f]) {
	  double *norm=normals_h+18*idx+3*(f-nccft_h[idx]);
	  facenorm_h.push_back(norm[0]);
	  facenorm_h.push_back(norm[1]);
	  facenorm_h.push_back(norm[2]);	  
	  cell2face_h[f]=nfaces+1;
	  int idxn=cell2cell[f];
	  if (idxn > -1) {
	    // make face info 1 based to use negative sign
	    int f1;
	    for(f1=nccft_h[idxn];f1<nccft_h[idxn+1] && cell2cell[f1]!=idx;f1++);
	    cell2face_h[f1]=-cell2face_h[f];
	    facetype_h.push_back(0);
	    iflag[f]=iflag[f1]=0;
	  }
	  else {
	    facetype_h.push_back(idxn);
	  }
	  nfaces++;
	}
      }
  printf("nfaces=%d\n",nfaces);

  facetype_d=gpu::push_to_device<int>(facetype_h.data(),sizeof(int)*nfaces);
  cell2face_d=gpu::push_to_device<int>(cell2face_h,sizeof(int)*cell2cell.size());
  facenorm_d=gpu::push_to_device<double>(facenorm_h.data(),sizeof(double)*nfaces*3);
 
  delete [] cell2face_h;
  delete [] normals_h;
}

void LocalMesh::InitSolution(double *flovar, int nfields)
{
 FVSAND_NVTX_FUNCTION( "init_solution" );
 FVSAND_NVTX_SECTION( "memory_allocation",
  q=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  qn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  qnn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  res_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  dq_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  dqupdate_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
 );
 
 flovar_d=gpu::push_to_device<double>(flovar,sizeof(double)*nfields);
 qinf_d=gpu::allocate_on_device<double>(sizeof(double)*nfields);
 
 nfields_d=nfields;
 nthreads=ncells+nhalo;
 //nthreads=ncells;
 n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
 FVSAND_GPU_LAUNCH_FUNC(init_q,n_blocks,block_size,0,0,
		        qinf_d,q,dq_d,center_d,flovar_d,nfields,istor,(ncells+nhalo));

 qh = new double[(ncells+nhalo)*nfields];
 gpu::pull_from_device<double>(qh,q,sizeof(double)*(ncells+nhalo)*nfields);

 int scale=(istor==0)?nfields:1;
 int stride=(istor==0)?1:ncells;
  
 for(auto s: sndmap) 
    for (auto v : s.second)
      for(int n=0;n<nfields_d;n++)
	device2host.push_back(v*scale+n*stride);
  
  for(auto r: rcvmap) 
    for (auto v : r.second)
      for(int n=0;n<nfields;n++)
	host2device.push_back(v*scale+n*stride);

  device2host_d=gpu::push_to_device<int>(device2host.data(),sizeof(int)*device2host.size());
  host2device_d=gpu::push_to_device<int>(host2device.data(),sizeof(int)*host2device.size());
 
  int dsize=device2host.size();
  dsize=(dsize < host2device.size())?host2device.size():dsize;
  qbuf=new double [dsize];
  qbuf2=new double [dsize];
  qbuf_d=gpu::allocate_on_device<double>(sizeof(double)*dsize);
  qbuf_d2=gpu::allocate_on_device<double>(sizeof(double)*dsize);

  ireq=new MPI_Request [sndmap.size()+rcvmap.size()];
  istatus=new MPI_Status [sndmap.size()+rcvmap.size()];
  
  faceq_d=gpu::allocate_on_device<double>(sizeof(double)*nfaces*2*nfields_d);
  faceflux_d=gpu::allocate_on_device<double>(sizeof(double)*nfaces*nfields_d);
 
}

void LocalMesh::UpdateFringes(double *qh, double *qd)
{
  
  nthreads=device2host.size();

  if(nthreads == 0) return;
  
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateHost,n_blocks,block_size,0,0,
                         qbuf_d,qd,device2host_d,device2host.size());
  gpu::pull_from_device<double>(qbuf,qbuf_d,sizeof(double)*device2host.size());

  for(int i=0;i<device2host.size();i++)
    qh[device2host[i]]=qbuf[i];
  
  //gpu::pull_from_device<double>(qh,qd,sizeof(double)*nfields_d*(ncells+nhalo));
  pc.exchangeDataDouble(qh,nfields_d,(ncells+nhalo),istor,sndmap,rcvmap,mycomm);
  //gpu::copy_to_device<double>(qd,qh,sizeof(double)*nfields_d*(ncells+nhalo));
  
  for(int i=0;i<host2device.size();i++)
    qbuf[i]=qh[host2device[i]];

  gpu::copy_to_device(qbuf_d,qbuf,sizeof(double)*host2device.size());

  nthreads=host2device.size();
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateDevice,n_blocks,block_size,0,0,
			 qd,qbuf_d,host2device_d,host2device.size());
}


void LocalMesh::UpdateFringes(double *qd)
{
    nthreads=device2host.size();
    if(nthreads == 0) return;
    n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
    FVSAND_GPU_LAUNCH_FUNC(updateHost,n_blocks,block_size,0,0,
			   qbuf_d,qd,device2host_d,device2host.size());

    

    // separate sends and receives so that we can overlap comm and calculation
    // in the residual and iteration loops.
    int reqcount=0;
    pc.postRecvs_direct(qbuf2,nfields_d,rcvmap,ireq,mycomm,&reqcount);
    // with cuda-aware this pull is not required
    // but it doesn't work now
    gpu::pull_from_device<double>(qbuf,qbuf_d,sizeof(double)*device2host.size());
    pc.postSends_direct(qbuf,nfields_d,sndmap,ireq,mycomm,&reqcount);
    pc.finish_comm(reqcount,ireq,istatus);
    // same as above
    // not doing cuda-aware now
    gpu::copy_to_device(qbuf_d2,qbuf2,sizeof(double)*host2device.size());
    
    nthreads=host2device.size();
    n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
    FVSAND_GPU_LAUNCH_FUNC(updateDevice,n_blocks,block_size,0,0,
			   qd,qbuf_d2,host2device_d,host2device.size());
}
    

void LocalMesh::Residual(double *qv,int restype)
{
  FVSAND_NVTX_FUNCTION("residual");
  if (restype==0) {
    Residual_cell(qv);
  } else {
    Residual_face(qv);
  }
}
				
void LocalMesh::Residual_cell(double *qv)
{
  nthreads=ncells;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(computeResidual,n_blocks,block_size,0,0,
			 res_d, qv, center_d, normals_d, volume_d,
			 qinf_d, cell2cell_d, nccft_d, nfields_d,istor,ncells);

  //UpdateFringes(qh,res_d);
  UpdateFringes(res_d);
  //parallelComm pc;
  //pc.exchangeDataDouble(qh,nfields_d,(ncells+nhalo),istor,sndmap,rcvmap,mycomm);

}

void LocalMesh::Residual_face(double *qv)
{
  nthreads=ncells+nhalo;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(fill_faces, n_blocks,block_size,0,0,
			 qv,faceq_d,nccft_d,cell2face_d,nfields_d,istor,ncells+nhalo);

  nthreads=nfaces;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(face_flux,n_blocks,block_size,0,0,
			 faceflux_d,faceq_d,facenorm_d,qinf_d,facetype_d,nfields_d,nfaces);

  nthreads=ncells;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(computeResidualFace,n_blocks,block_size,0,0,
			 res_d,faceflux_d,volume_d,cell2face_d,nccft_d,
			 nfields_d,istor,ncells);
  //UpdateFringes(qh,res_d);
  UpdateFringes(res_d);

}

void LocalMesh::Jacobi(double *q, double dt, int nsweep, int istoreJac)
{
/*  FVSAND_GPU_LAUNCH_FUNC(testComputeJ,n_blocks,block_size,0,0,
    q,normals_d,flovar_d, cell2cell_d,nccft_d,nfields_d,istor,ncells,facetype_d);
    exit(0);
  */
  nthreads=(ncells+nhalo)*nfields_d;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(setValues,n_blocks,block_size,0,0,
			 dq_d, 0.0, (ncells+nhalo)*nfields_d);
  //compute and store Jacobians;
  if(istoreJac){
    FVSAND_GPU_LAUNCH_FUNC(fillJacobians,n_blocks,block_size,0,0,
			   q, normals_d, volume_d,
			   rmatall_d, Dall_d,
			   flovar_d, cell2cell_d,
			   nccft_d, nfields_d, istor, ncells, facetype_d, dt);
  }
  // Jacobi Sweeps
  for(int m = 0; m < nsweep; m++){
    //printf("Sweep %i\n=================\n",m);
    nthreads=ncells+nhalo;
    n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);

    // compute dqtilde for all cells
    if(istoreJac){
      FVSAND_GPU_LAUNCH_FUNC(jacobiSweep2,n_blocks,block_size,0,0,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     rmatall_d, Dall_d,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, istor, ncells, facetype_d, dt);
    }
    else{
      FVSAND_GPU_LAUNCH_FUNC(jacobiSweep,n_blocks,block_size,0,0,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, istor, ncells, facetype_d, dt);
    }
    // update dq = dqtilde for all cells
    nthreads=(ncells+nhalo)*nfields_d;
    n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
    FVSAND_GPU_LAUNCH_FUNC(copyValues,n_blocks,block_size,0,0,
			   dq_d, dqupdate_d, (ncells+nhalo)*nfields_d);

    // Store final dq in res to be used in update routine
    UpdateFringes(dq_d);
    //UpdateFringes(qh,dq_d);
  }
}

void LocalMesh::Update(double *qdest, double *qsrc, double fscal)
{
  FVSAND_NVTX_FUNCTION( "update" );
  nthreads=(ncells+nhalo)*nfields_d;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateFields,n_blocks,block_size,0,0,
			 res_d, qdest, qsrc, fscal, (ncells+nhalo)*nfields_d);
}

void LocalMesh::UpdateQ(double *qdest, double *qsrc, double fscal)
{
  nthreads=(ncells+nhalo)*nfields_d;
  n_blocks=nthreads/block_size + (nthreads%block_size==0 ? 0:1);
  FVSAND_GPU_LAUNCH_FUNC(updateFields,n_blocks,block_size,0,0,
			 dq_d, qdest, qsrc, fscal, (ncells+nhalo)*nfields_d);
}

double LocalMesh::ResNorm()
{
  FVSAND_NVTX_FUNCTION( "ResNorm" );

  gpu::pull_from_device<double>(qh,res_d,sizeof(double)*nfields_d*(ncells+nhalo));
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

  gpu::pull_from_device<double>(qh,volume_d,sizeof(double)*(ncells+nhalo));
  for(i=0;i<ncells+nhalo;i++)
    fprintf(fp,"%lf\n",qh[i]);

  gpu::pull_from_device<double>(qh,q,sizeof(double)*nfields_d*(ncells+nhalo));
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
