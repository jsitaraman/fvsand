#include "pyfv.h"
#include "fvsand_gpu.h"
#include "inputParser.h"

using namespace std;
using namespace FVSAND;
namespace py = pybind11;

#define DO_NOT_FREE py::capsule([](){})

PyFV::PyFV(string inputfile)
{

  this->meshtype=0;
  this->dsmin=0.01;
  this->stretch=1.1;
  this->nlevels=30;
  this->nfields=5;
  this->flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4, 100};
  this->nsteps=2000;
  this->nsave=100;
  this->dt=0.03;
  this->reOrderCells=false; // Re-order cells for better cache efficiency 
  this->nsweep = 2;   // Jacobi Sweeps (=0 means explict)
  this->istoreJac =3; // Jacobian storage or not 
  this->restype=0;    // restype = 0 (cell-based) 1 (face-based)
  this->nsubit=10;


  char fname[]="data.tri";

  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

#if FVSAND_HAS_GPU
  FVSAND_GPU_CHECK_ERROR(cudaGetDeviceCount(&numdevices));
  mydeviceid = myid % numdevices;
  FVSAND_GPU_CHECK_ERROR(cudaSetDevice(mydeviceid));
  gpu::listdev(myid);
#endif

  parseInputs(inputfile.c_str(),&meshtype,fname,&dsmin,&stretch,&nlevels,
              flovar,&nsteps,&nsave,&dt,reOrderCells,&nsweep,&nsubit,
	      &istoreJac,&restype);

  // create cartesian or strand mesh
  //
  if (meshtype==1) {
    CartesianMesh *cm;
    cm= new CartesianMesh(fname,numprocs);
    cm->WriteMesh(myid);
    lm= new LocalMesh(cm,myid,MPI_COMM_WORLD);
    ncells=cm->ncells;
  }
  else
  {
    StrandMesh *sm;
    sm=new StrandMesh(fname,dsmin,stretch,nlevels,myid);
    if (reOrderCells) sm->ReOrderCells();
    sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);      
    // create local mesh partitions
    // and compute grid metrics
    lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
 
    ncells=sm->ncells;
  }
  lm->CreateGridMetrics(istoreJac);
  // initialize solution
  lm->InitSolution(flovar.data(),nfields);
}

PyFV::~PyFV()
{
  double elapsed = stopwatch.elapsed();
  if (myid == 0) 
  {
    printf("# ----------------------------------\n");
    printf("# Elapsed time: %13.4f s\n", elapsed);
    printf("# Through-put : %13.4f [million-elements/sec/iteration]\n",
           ncells/(elapsed/nsteps)/1e6);
    printf("# ----------------------------------\n");
  }

  lm->WriteMesh(myid);  

  // delete sm;
  // delete lm;
}

void PyFV::step(int iter)
{

  stopwatch.tick();

  
  std::ostringstream timestep_name;
  timestep_name << "TimeStep-" << iter;
  if(nsweep){ // implicit 
    //FVSAND_NVTX_SECTION(timestep_name.str(),
    if (nsubit > 1) lm->update_time();
    for(int it=0;it < nsubit;it++) {
      lm->Residual(lm->q,restype,dt,istoreJac); // computes res_d
      if (nsubit > 1) lm->add_time_source(iter,dt, lm->q,lm->qn,lm->qnn);
      lm->Jacobi(lm->q,dt,nsweep,istoreJac); // runs sweeps and replaces res_d with
      //if (nsubit > 1) lm->RegulateDQ(lm->q);
      lm->UpdateQ(lm->q,lm->q,1);            // adds dqtilde (in res_d)
      if (nsubit > 1) {
        double rnorm=lm->ResNorm(lm->res_d);
        if (myid==0) printf("%6d %6d  %16.8e\n",iter, it,rnorm);
      }
    }
    //);
  }else {
    FVSAND_NVTX_SECTION( timestep_name.str(), 
                         lm->Residual(lm->q,restype);
                         lm->Update(lm->qn,lm->q,rk[1]*dt);
                         lm->Update(lm->q,lm->q,rk[0]*dt);
                         
                         lm->Residual(lm->qn,restype);
                         lm->Update(lm->qn,lm->q,rk[2]*dt);
                         
                         lm->Residual(lm->qn,restype);      
                         lm->Update(lm->q,lm->q,rk[3]*dt);
                         );
  }
  if (nsubit==1) {
    if ((iter+1)%nsave ==0 || iter==0) {
      double rnorm=lm->ResNorm(lm->res_d);
      if (myid==0) printf("iter:%6d  %16.8e\n",iter+1,rnorm);
    }
  }

  stopwatch.tock();
}


// self.gridData={'gridtype':'unstructured',        
//                'tetConn':self.ndc4,              
//                'pyraConn':self.ndc5,             
//                'prismConn':self.ndc6,            
//                'hexaConn':self.ndc8,             
//                'bodyTag':self.tag,               
//                'wallnode':self.inbcout,          
//                'obcnode':self.iobcout,           
//                'grid-coordinates':self.xout,                  
//                'q-variables':self.qout,          
//                'iblanking':self.ibout,           
//                'iblkHasNBHole':0,                
//                'istor':'row',       
//                'scaling':self.scale,             
//                'mapping':self.mapping,           
//                'fsitag':self.fsitag,             
//                'fsinode':self.fsinode,           
//                'fsiforce':self.fsiforce,         
//                'fsipforce':self.fsipforce,       
//                'fsicoord':self.fsicoord} 

py::dict PyFV::get_grid_data(){

  py::dict gridData;
  // lists of stuff to pass back to python
  py::list ibl(1);
  py::list ql(1);
  py::list xl(1);
  py::list xl_d(1);
  py::list c2nl(1);
  py::list c2nl_d(1);
  py::list c2cl(1);
  py::list c2cl_d(1);
  py::list nvl(1);
  py::list nvl_d(1);
  py::list ncl(1);
  py::list ncl_d(1);
  py::list obcnode(1);
  py::list obcnode_d(1);
  py::list wbcnode(1);
  py::list wbcnode_d(1);

  double *x_hd[2];
  int nnode, ncell, nc2n,nc2c,nobc,nwbc;
  // int *ndc4, *ndc5, *ndc6, *ndc8;
  int* cell2node_hd[2];
  int *cell2cell_hd[2];
  int *nccft_hd[2];
  int *obcnode_hd[2];
  int *wbcnode_hd[2];
  int* nvcft_hd[2];

  lm->GetGridData(x_hd, &nnode, &ncell, nvcft_hd, nccft_hd,
		  	cell2node_hd, &nc2n, 
		  	cell2cell_hd, &nc2c, 
			obcnode_hd, &nobc, 
			wbcnode_hd, &nwbc);

  c2nl[0]   = py::array_t<int   >({nc2n},          {sizeof(int)},    cell2node_hd[0], DO_NOT_FREE);
  c2cl[0]   = py::array_t<int   >({nc2c},          {sizeof(int)},    cell2cell_hd[0], DO_NOT_FREE);
  nvl[0]    = py::array_t<int   >({ncell+1},       {sizeof(int)},    nvcft_hd[0],     DO_NOT_FREE);
  ncl[0]    = py::array_t<int   >({ncell+1},       {sizeof(int)},    nccft_hd[0],     DO_NOT_FREE);
  xl[0]     = py::array_t<double>({nnode*3},       {sizeof(double)}, x_hd[0],         DO_NOT_FREE);
  obcnode[0]= py::array_t<int >  ({nobc},          {sizeof(int)},    obcnode_hd[0],   DO_NOT_FREE);
  wbcnode[0]= py::array_t<int >  ({nwbc},          {sizeof(int)},    wbcnode_hd[0],   DO_NOT_FREE); 

#ifdef FVSAND_HAS_CUDA
  ibl[0]    = GPUArray(lm->iblank, ncell);
  ql[0]     = GPUArray(lm->q,      nfields*ncell);
  c2nl_d[0] = GPUArray(cell2node_hd[1], nc2n);
  nvl_d[0]  = GPUArray(nvcft_hd[1], ncell+1);
  ncl_d[0]  = GPUArray(nccft_hd[1], ncell+1);
  xl_d[0]   = GPUArray(x_hd[1], nnode*3);
  c2cl_d[0]  = GPUArray(cell2cell_hd[1],nc2c);
  obcnode_d[0]= GPUArray(nccft_hd[1],ncell+1);
  wbcnode_d[0]= GPUArray(wbcnode_hd[1],nwbc+1);
  gridData["memtype"] = "CUDA";
#else     
  ibl[0]    = py::array_t<int   >({ncell},         {sizeof(int)},    lm->iblank,      DO_NOT_FREE);
  ql[0]     = py::array_t<double>({ncell*nfields}, {sizeof(double)}, lm->q,           DO_NOT_FREE);
  c2nl_d[0] = c2nl[0];
  nvl_d[0]  = nvl[0];
  xl_d[0]   = xl[0];
  c2cl_d[0]  = c2cl[0];
  obcnode_d[0] = obcnode[0];
  wbcnode_d[0] = wbcnode[0];
  ncl_d[0]     = ncl[0];
  gridData["memtype"] = "CPU";
#endif

  gridData["ncell"]              = ncell;
  gridData["nnode"]              = nnode;
  gridData["nobc"]               = nobc;
  gridData["nwbc"]               = nwbc;
  gridData["iblanking"]          = ibl;
  gridData["q-variables"]        = ql;
  gridData["grid-coordinates"]   = xl;
  gridData["grid-coordinates_d"] = xl_d;
  gridData["cell2node"]          = c2nl;
  gridData["cell2node_d"]        = c2nl_d;
  gridData["nvcft"]              = nvl;
  gridData["nvcft_d"]            = nvl_d;
  gridData["cell2cell"]          = c2cl;
  gridData["cell2cell_d"]        = c2cl_d;
  gridData["nccft"]              = ncl;
  gridData["nccft_d"]            = ncl_d;
  gridData["obcnode"]            = obcnode;
  gridData["obcnode_d"]          = obcnode_d;
  gridData["wbcnode"]            = wbcnode;
  gridData["wbcnode_d"]          = wbcnode_d;

  return gridData;

}
