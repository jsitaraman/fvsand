#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "fvsand_gpu.h"
#include "metric_functions.h"
#include "solver_functions.h"
#include "NVTXMacros.h"
#include "timer.h"
#include <cstdio>

#if defined(FVSAND_HAS_GPU) && defined(FVSAND_HAS_CUDA) && !defined(FVSAND_FAKE_GPU)
// add CUDA thrust includes
#include <thrust/transform_reduce.h>
#endif

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
  FVSAND_FREE_DEVICE(centroid_d);
  FVSAND_FREE_DEVICE(facecentroid_d);
  FVSAND_FREE_DEVICE(normals_d);
  FVSAND_FREE_DEVICE(volume_d);
  FVSAND_FREE_DEVICE(res_d);
  FVSAND_FREE_DEVICE(grad_d);
  FVSAND_FREE_DEVICE(gradweights_d);
  //
  FVSAND_FREE_DEVICE(rmatall_d);
  FVSAND_FREE_DEVICE(Dall_d);
  FVSAND_FREE_DEVICE(rmatall_d_f);
  FVSAND_FREE_DEVICE(Dall_d_f);
  //
  FVSAND_FREE_DEVICE(cell2face_d);
  FVSAND_FREE_DEVICE(face2cell_d);
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
  Timer stopwatch;
  mycomm=comm;
  int ierr=MPI_Comm_rank(comm,&myid);
  ierr=MPI_Comm_size(comm,&ngroup);
  parallelComm pc;
  
  // create communication patterns and ghost cells
  stopwatch.tick(); 
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
			rcvmap,mycomm);
  double elapsed=stopwatch.tock();
  //printf("Comm Patterns time %e\n",elapsed);
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

void LocalMesh::CreateGridMetrics(int istoreJac)
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
  //
  if (istoreJac==1 || istoreJac ==4) {
    Dall_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*25);
  }
  if (istoreJac==5 || istoreJac==6 || istoreJac==7 || istoreJac==8 || istoreJac==9 ||
      istoreJac==10 || istoreJac==11) {
    Dall_d_f=gpu::allocate_on_device<float>(sizeof(float)*(ncells+nhalo)*25);
  }
  if (istoreJac==1) {
    rmatall_d=gpu::allocate_on_device<double>(sizeof(double)*nccft_h[ncells+nhalo]*25);
  }
  if (istoreJac==6 || istoreJac==8) {
    rmatall_d_f=gpu::allocate_on_device<float>(sizeof(float)*nccft_h[ncells+nhalo]*25);
  }
  // allocate storage for metrics
  center_d=gpu::allocate_on_device<double>(sizeof(double)*3*(ncells+nhalo));
  centroid_d=gpu::allocate_on_device<double>(sizeof(double)*3*(ncells+nhalo));
  facecentroid_d=gpu::allocate_on_device<double>(sizeof(double)*18*(ncells+nhalo));
  // allocate larger storage than necessary for normals to avoid
  // unequal stride, 6=max faces for hex and 3 doubles per face
  normals_d=gpu::allocate_on_device<double>(sizeof(double)*18*(ncells+nhalo));
  volume_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo));
  gradweights_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells)*18);
  // compute cell center_ds

  int N = ncells + nhalo;
  FVSAND_GPU_KERNEL_LAUNCH( cell_center, (N*3), center_d, x_d, nvcft_d, 
                            cell2node_d, N );

  // compute cell normals, centroid and volume_d
  FVSAND_GPU_KERNEL_LAUNCH( cell_normals_volume, N, normals_d, volume_d, centroid_d, facecentroid_d,
			    x_d, ncon_d, cell2node_d, nvcft_d, N );

  // check conservation
  FVSAND_GPU_KERNEL_LAUNCH( check_conservation, N, normals_d, x_d, nccft_d, 
                            cell2cell_d, N );

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
  std::vector< std::array<int,3> > face2cell;
  double *normals_h = new double [(ncells+nhalo)*18];
  gpu::pull_from_device<double>(normals_h,normals_d,sizeof(double)*(ncells+nhalo)*18);

  nfaces=0;
  for(int idx=0;idx<ncells;idx++)
    for(int f=nccft_h[idx];f<nccft_h[idx+1];f++)
      {
	if (iflag[f]) {
	  double norm[3];
	  for(int d=0;d<3;d++)
	    norm[d]=normals_h[(3*(f-nccft_h[idx])+d)*(ncells+nhalo)+idx];
	  facenorm_h.push_back(norm[0]);
	  facenorm_h.push_back(norm[1]);
	  facenorm_h.push_back(norm[2]);	  
	  cell2face_h[f]=nfaces+1;
	  int idxn=cell2cell[f];
          std::array<int,3> arr = {idx,idxn,f};
          face2cell.emplace_back(arr);
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
  //printf("nfaces=%d\n",nfaces);
  //
  int *face2cell_h=new int [3*nfaces];
  for(int f=0;f<nfaces;f++) {
    face2cell_h[f] = face2cell[f][0];
    face2cell_h[nfaces+f] = face2cell[f][1];
    face2cell_h[2*nfaces+f] = face2cell[f][2];
  }

  face2cell.clear();

  facetype_d=gpu::push_to_device<int>(facetype_h.data(),sizeof(int)*nfaces);
  cell2face_d=gpu::push_to_device<int>(cell2face_h,sizeof(int)*cell2cell.size());
  face2cell_d=gpu::push_to_device<int>(face2cell_h,sizeof(int)*nfaces*3);
  facenorm_d=gpu::push_to_device<double>(facenorm_h.data(),sizeof(double)*nfaces*3);
 
  delete [] cell2face_h;
  delete [] face2cell_h;
  delete [] normals_h;
}

void LocalMesh::InitSolution(double *flovar, int nfields)
{
 FVSAND_NVTX_FUNCTION( "init_solution" );
 // FVSAND_NVTX_SECTION( "memory_allocation",
  q=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  qn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  qnn=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  res_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  dq_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  dqupdate_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields);
  grad_d=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields*4);
  // );
 
 flovar_d=gpu::push_to_device<double>(flovar,sizeof(double)*nfields);
 qinf_d=gpu::allocate_on_device<double>(sizeof(double)*nfields);


 scale=(istor==0)?nfields:1;
 stride=(istor==0)?1:(ncells+nhalo);

 nfields_d=nfields;
 int N=ncells+nhalo;
 FVSAND_GPU_KERNEL_LAUNCH( init_q, N, 
			   qinf_d,q,dq_d,center_d,flovar_d,nfields,scale,stride,N);
 
 N=ncells;
 FVSAND_GPU_KERNEL_LAUNCH( weighted_least_squares, N, gradweights_d,
			   centroid_d, facecentroid_d,
			   cell2cell_d, nccft_d,
			   scale, stride, N);
 
 qh = new double[(ncells+nhalo)*nfields];
 gpu::pull_from_device<double>(qh,q,sizeof(double)*(ncells+nhalo)*nfields);

 
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
// Old inefficient update fringes, just preserved here for
// posterity
void LocalMesh::UpdateFringes(double *qh, double *qd)
{
  
  nthreads=device2host.size();
  if(nthreads == 0) return;
  FVSAND_GPU_KERNEL_LAUNCH( updateHost, nthreads,
			    qbuf_d,qd,device2host_d,nthreads);
  
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
  FVSAND_GPU_KERNEL_LAUNCH( updateDevice, nthreads,
			 qd,qbuf_d,host2device_d,host2device.size());

}

// New update with persistent buffers and minimum copies
// doesn't work CUDA-Aware -- debug this
void LocalMesh::UpdateFringes(double *qd)
{
    nthreads=device2host.size();
    if(nthreads == 0) return;
    FVSAND_GPU_KERNEL_LAUNCH( updateHost, nthreads,
			      qbuf_d,qd,device2host_d,nthreads);
    // separate sends and receives so that we can overlap comm and calculation
    // in the residual and iteration loops.
    // TODO (george) use qbuf2_d and qbuf_d instead of qbuf2 and qbuf for cuda-aware
    int reqcount=0;
    pc.postRecvs_direct(qbuf2,nfields_d,rcvmap,ireq,mycomm,&reqcount);
    // TODO (george) with cuda-aware this pull is not required
    // but it doesn't work now
    gpu::pull_from_device<double>(qbuf,qbuf_d,sizeof(double)*device2host.size());
    pc.postSends_direct(qbuf,nfields_d,sndmap,ireq,mycomm,&reqcount);
    pc.finish_comm(reqcount,ireq,istatus);
    // same as above
    // not doing cuda-aware now
    gpu::copy_to_device(qbuf_d2,qbuf2,sizeof(double)*host2device.size());
    
    nthreads=host2device.size();
    FVSAND_GPU_KERNEL_LAUNCH( updateDevice, nthreads,
			      qd,qbuf_d2,host2device_d,nthreads);
}
    

void LocalMesh::Residual(double *qv, int restype, double dt, int istoreJac)
{
  FVSAND_NVTX_FUNCTION("residual");
  if ( istoreJac == 6 ) {
    Residual_Jacobian(qv,dt);
    return;
  }

  if ( istoreJac == 7 ) {
    Residual_Jacobian_diag(qv,dt);
    return;
  }

  if ( istoreJac == 10 ) {
    Residual_Jacobian_diag_face(qv,dt);
    return;
  }

  if ( istoreJac == 11 ) {
    Residual_Jacobian_diag_face2(qv,dt);
    return;
  }

  if (restype==0) {
    Residual_cell(qv);
  } else if (restype==1) {
    Residual_face(qv);
  }
}
				
void LocalMesh::Residual_cell(double *qv)
{
  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidual,nthreads,
			   res_d, qv, center_d, normals_d, volume_d,
			   qinf_d, cell2cell_d, nccft_d, nfields_d,scale,stride,ncells);
  UpdateFringes(res_d);

}

void LocalMesh::Residual_face(double *qv)
{
  nthreads=ncells+nhalo;
  FVSAND_GPU_KERNEL_LAUNCH(fill_faces,nthreads,
			   qv,faceq_d,nccft_d,cell2face_d,nfields_d,scale, stride,ncells+nhalo);

  nthreads=nfaces;
  FVSAND_GPU_KERNEL_LAUNCH(face_flux,nthreads,
			   faceflux_d,faceq_d,facenorm_d,qinf_d,facetype_d,nfields_d,nfaces);

  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidualFace,nthreads,
			   res_d,faceflux_d,volume_d,cell2face_d,nccft_d,
			   nfields_d,scale,stride,ncells);
  UpdateFringes(res_d);
}


void LocalMesh::Residual_Jacobian(double *qv, double dt)
{
  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidualJacobian,nthreads,
        		   qv, normals_d, volume_d,
        		   res_d, rmatall_d_f, Dall_d_f,
        		   qinf_d, cell2cell_d,
                           nccft_d, nfields_d, scale, stride, ncells, dt);
  UpdateFringes(res_d);
}

void LocalMesh::Residual_Jacobian_diag(double *qv, double dt)
{
  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidualJacobianDiag,nthreads,
        		   qv, normals_d, volume_d,
        		   res_d, Dall_d_f,
        		   qinf_d, cell2cell_d,
                           nccft_d, nfields_d, scale, stride, ncells, dt);
  UpdateFringes(res_d);
}

void LocalMesh::Residual_Jacobian_diag_face(double *qv, double dt)
{
  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(setResidual,nthreads,
			   res_d, qv, center_d, normals_d, volume_d,
			   qinf_d, cell2cell_d, nccft_d, nfields_d,scale,stride,ncells);

  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(setJacobians_diag_f,nthreads,
			     qv, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  nthreads=nfaces;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidualJacobianDiagFace,nthreads,
        		   qv, normals_d, volume_d,
        		   res_d, Dall_d_f,
        		   qinf_d, face2cell_d,
                           nccft_d, nfields_d, scale, stride, ncells, nfaces, dt);

  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(scaleResidual,nthreads,
			   res_d, qv, center_d, normals_d, volume_d,
			   qinf_d, cell2cell_d, nccft_d, nfields_d,scale,stride,ncells);
  UpdateFringes(res_d);
}

void LocalMesh::Residual_Jacobian_diag_face2(double *qv, double dt)
{
  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(setResidual,nthreads,
			   res_d, qv, center_d, normals_d, volume_d,
			   qinf_d, cell2cell_d, nccft_d, nfields_d,scale,stride,ncells);

  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(setJacobians_diag_f,nthreads,
			     qv, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  nthreads=nfaces;
  FVSAND_GPU_KERNEL_LAUNCH(computeResidualJacobianDiagFace2,nthreads,
        		   qv, normals_d, volume_d,
        		   res_d, Dall_d_f,
        		   qinf_d, face2cell_d,
                           nccft_d, nfields_d, scale, stride, ncells, nfaces, dt);

  nthreads=ncells;
  FVSAND_GPU_KERNEL_LAUNCH(scaleResidual,nthreads,
			   res_d, qv, center_d, normals_d, volume_d,
			   qinf_d, cell2cell_d, nccft_d, nfields_d,scale,stride,ncells);
  UpdateFringes(res_d);
}

void LocalMesh::Jacobi(double *q, double dt, int nsweep, int istoreJac)
{
/*  FVSAND_GPU_LAUNCH_FUNC(testComputeJ,n_blocks,block_size,0,0,
    q,normals_d,flovar_d, cell2cell_d,nccft_d,nfields_d,istor,ncells,facetype_d);
    exit(0);
  */
  FVSAND_NVTX_FUNCTION("Jacobi");
  nthreads=(ncells+nhalo)*nfields_d;
  FVSAND_GPU_KERNEL_LAUNCH(setValues,nthreads,
			 dq_d, 0.0, nthreads);
  //compute and store Jacobians;
  if(istoreJac==1){
    nthreads=ncells+nhalo;
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobians,nthreads,
			   q, normals_d, volume_d,
			   rmatall_d, Dall_d,
			   flovar_d, cell2cell_d,
			   nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  }
  if (istoreJac==4) {
    nthreads=ncells+nhalo;
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobians_diag,nthreads,
                           q, normals_d, volume_d,
			     Dall_d,
                           flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  }
  if (istoreJac==5) {
    nthreads=ncells+nhalo;
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobians_diag_f,nthreads,
			     q, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  }
  if (istoreJac==8) {
    nthreads=ncells+nhalo;
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobians_diag_f,nthreads,
			     q, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobians_offdiag_f,nthreads,
			     q, normals_d, volume_d,
			     rmatall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
  }
  if (istoreJac==9) {
    nthreads=ncells;
    FVSAND_GPU_KERNEL_LAUNCH(setJacobians_diag_f,nthreads,
			     q, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    nthreads=nfaces;
    FVSAND_GPU_KERNEL_LAUNCH(fillJacobiansFace_diag_f,nthreads,
			     q, normals_d, volume_d,
			     Dall_d_f,
			     flovar_d, face2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, nfaces, dt);
  }
  
  // Jacobi Sweeps
  for(int m = 0; m < nsweep; m++){
    //printf("Sweep %i\n=================\n",m);
    nthreads=ncells+nhalo;
    // compute dqtilde for all cells
    if(istoreJac==0){
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep,nthreads,
			       q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			       flovar_d, cell2cell_d,
			       nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    }
    else if(istoreJac==1) {
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep1,nthreads,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     rmatall_d, Dall_d,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);      
    }
    else if(istoreJac==2) {
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep2,nthreads,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale,stride, ncells, facetype_d, dt);
    }
    else if(istoreJac==3) {
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep3,nthreads,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     flovar_d, cell2cell_d,
			       nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    }
    else if(istoreJac==4) {
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep4,nthreads,
			       q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			       Dall_d,
			       flovar_d, cell2cell_d,
			       nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    }
    else if(istoreJac==5 || istoreJac==7 || istoreJac==9 || istoreJac==10 || istoreJac==11) {
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep5,nthreads,
			       q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			       Dall_d_f,
			       flovar_d, cell2cell_d,
			       nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);
    }
    else if(istoreJac==6 || istoreJac==8){
      FVSAND_GPU_KERNEL_LAUNCH(jacobiSweep1_f,nthreads,
			     q, res_d, dq_d, dqupdate_d, normals_d, volume_d,
			     rmatall_d_f, Dall_d_f,
			     flovar_d, cell2cell_d,
			     nccft_d, nfields_d, scale, stride, ncells, facetype_d, dt);      
    }
    // update dq = dqtilde for all cells
    
    nthreads=(ncells+nhalo)*nfields_d;
    FVSAND_GPU_KERNEL_LAUNCH(copyValues,nthreads,
			   dq_d, dqupdate_d, nthreads);
   
    // Store final dq in res to be used in update routine
    UpdateFringes(dq_d);
  }
}

void LocalMesh::Update(double *qdest, double *qsrc, double fscal)
{
  FVSAND_NVTX_FUNCTION( "update" );
  nthreads=(ncells+nhalo)*nfields_d;
  FVSAND_GPU_KERNEL_LAUNCH(updateFields,nthreads,
			 res_d, qdest, qsrc, fscal, nthreads);
}

void LocalMesh::UpdateQ(double *qdest, double *qsrc, double fscal)
{
  nthreads=(ncells+nhalo)*nfields_d;
  FVSAND_GPU_KERNEL_LAUNCH(updateFields,nthreads,
			 dq_d, qdest, qsrc, fscal, nthreads);
}

#if defined(FVSAND_HAS_GPU) && defined(FVSAND_HAS_CUDA) && !defined(FVSAND_FAKE_GPU)
template <typename T>
struct fvsand_square_op
{
  __host__ __device__
  T operator()(const T& x) const { 
      return x * x;
  }
};
#endif

double LocalMesh::ResNorm()
{
  FVSAND_NVTX_FUNCTION( "ResNorm" );

  double rnorm[2];
  double rnormTotal[2];
  rnorm[0]=0.0;
  rnorm[1]=ncells;

#if defined(FVSAND_HAS_GPU) && defined(FVSAND_HAS_CUDA) && !defined(FVSAND_FAKE_GPU)
  fvsand_square_op<double> unary_op;
  thrust::plus<double> binary_op;
  const int N = nfields_d * ncells; 
  rnorm[ 0 ]  = thrust::transform_reduce( thrust::device
                                        , res_d
                                        , res_d+N
                                        , unary_op
                                        , 0.0
                                        , binary_op );
#else
  gpu::pull_from_device<double>(qh,res_d,sizeof(double)*nfields_d*(ncells+nhalo));
  for(int i=0;i<nfields_d*ncells;i++)
    rnorm[0]+=(qh[i]*qh[i]);
#endif

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
