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
  this->flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4};
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

  // create strand mesh
  sm=new StrandMesh(fname,dsmin,stretch,nlevels,myid);
  if (reOrderCells) sm->ReOrderCells();
  sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);

  lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
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
           sm->ncells/(elapsed/nsteps)/1e6);
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
  py::list nvl(1);
  py::list nvl_d(1);

  double *x_hd[2];
  int nnode, ncell, nc2n;
  // int *ndc4, *ndc5, *ndc6, *ndc8;
  int* cell2node_hd[2];
  int* nvcft_hd[2];

  lm->GetGridData(x_hd, &nnode, &ncell, nvcft_hd, cell2node_hd, &nc2n);

  c2nl[0]   = py::array_t<int   >({nc2n},          {sizeof(int)},    cell2node_hd[0], DO_NOT_FREE);
  nvl[0]    = py::array_t<int   >({ncell+1},       {sizeof(int)},    nvcft_hd[0],     DO_NOT_FREE);
  xl[0]     = py::array_t<double>({nnode*3},       {sizeof(double)}, x_hd[0],         DO_NOT_FREE);

#ifdef FVSAND_HAS_CUDA
  ibl[0]    = GPUArray(lm->iblank, ncell);
  ql[0]     = GPUArray(lm->q,      nfields*ncell);
  c2nl_d[0] = GPUArray(cell2node_hd[1], nc2n);
  nvl_d[0]  = GPUArray(nvcft_hd[1], ncell+1);
  xl_d[0]   = GPUArray(x_hd[1], nnode*3);
  gridData["memtype"] = "CUDA";
#else     
  ibl[0]    = py::array_t<int   >({ncell},         {sizeof(int)},    lm->iblank,      DO_NOT_FREE);
  ql[0]     = py::array_t<double>({ncell*nfields}, {sizeof(double)}, lm->q,           DO_NOT_FREE);
  c2nl_d[0] = c2nl[0];
  nvl_d[0]  = nvl[0];
  xl_d[0]   = xl[0];
  gridData["memtype"] = "CPU";
#endif

  gridData["ncell"]              = ncell;
  gridData["nnode"]              = nnode;
  gridData["iblanking"]          = ibl;
  gridData["q-variables"]        = ql;
  gridData["grid-coordinates"]   = xl;
  gridData["grid-coordinates_d"] = xl_d;
  gridData["cell2node"]          = c2nl;
  gridData["cell2node_d"]        = c2nl_d;
  gridData["nvcft"]              = nvl;
  gridData["nvcft_d"]            = nvl_d;

  return gridData;

}
