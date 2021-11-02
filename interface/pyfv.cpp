#include "pyfv.h"
#include "fvsand_gpu.h"
#include "inputParser.h"

using namespace std;
using namespace FVSAND;
namespace py = pybind11;

#define DO_NOT_FREE py::capsule([](){})

PyFV::PyFV(string inputfile)
{

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

  char fname[]="data.tri";

  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

#if FVSAND_HAS_GPU
  FVSAND_GPU_CHECK_ERROR(cudaGetDeviceCount(&numdevices));
  mydeviceid = myid % numdevices;
  FVSAND_GPU_CHECK_ERROR(cudaSetDevice(mydeviceid));
  gpu::listdev(myid);
#endif

  parseInputs(inputfile.c_str(),fname,&dsmin,&stretch,&nlevels,
	      flovar,&nsteps,&nsave,&dt,reOrderCells,&nsweep,
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
  if(nsweep)
  { // implicit 
    FVSAND_NVTX_SECTION( timestep_name.str(),
                         lm->Residual(lm->q,restype);           // computes res_d
                         lm->Jacobi(lm->q,dt,nsweep,istoreJac); // runs sweeps and replaces res_d with dqtilde
                         lm->UpdateQ(lm->q,lm->q,1);            // adds dqtilde (in res_d) to q XX is this dt or 1?
                         );
  } 
  else 
  {
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
  
  if ((iter+1)%nsave ==0 || iter==0) 
  {
    double rnorm=lm->ResNorm();
    if (myid==0) printf("iter:%6d  %16.8e\n",iter+1,rnorm);
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
  py::list l1, l2, l3, l4, l5;

  double *x;
  int nnode, ncell;
  int *ndc4, *ndc5, *ndc6, *ndc8;

  lm->GetGridData(&x, &nnode, &ncell, &ndc4, &ndc5, &ndc6, &ndc8);

  l1 = py::list(1); // lists of lengh # local patches, always 1
  l2 = py::list(1);

#ifdef FVSAND_HAS_CUDA
  l1[0] = GPUArray(lm->iblank, ncell);
  l2[0] = GPUArray(lm->q,      nfields*ncell);
#else
  l1[0] = py::array_t<int   >({ncell},         {sizeof(int)},    lm->iblank, DO_NOT_FREE);
  l2[0] = py::array_t<double>({ncell*nfields}, {sizeof(double)}, lm->q,      DO_NOT_FREE);
#endif

  gridData["iblanking"]   = l1;
  gridData["q-variables"] = l2;

  return gridData;

}
