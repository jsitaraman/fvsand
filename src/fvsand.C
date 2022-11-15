#include <iostream>
#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "fvsand_gpu.h"
#include "timer.h"
#include <typeinfo>
#include <bitset>
#include <string>
#include "NVTXMacros.h"
#include <sstream> // for std::ostringstream
#include "inputParser.h"
using namespace FVSAND;

// -----------------------------------------------------------------------------
#if FVSAND_HAS_GPU
#include "cuda_runtime.h"
void listdev( int rank )
{
    cudaError_t err;
    
    int dev_cnt = 0;
    cudaSetDevice(rank);
    err = cudaGetDeviceCount( &dev_cnt );
    assert( err == cudaSuccess || err == cudaErrorNoDevice );
    printf( "rank %d, cnt %d\n", rank, dev_cnt );
    
    cudaDeviceProp prop;
    for (int dev = 0; dev < dev_cnt; ++dev) {
        err = cudaGetDeviceProperties( &prop, dev );
        assert( err == cudaSuccess );

        // printf( "rank %d, dev %d, prop %s, pci %d, %d, %d\n",
        //         rank, dev,
        //         prop.name,
        //         prop.pciBusID,
        //         prop.pciDeviceID,
        //         prop.pciDomainID );

        // dylan: for NVLINK systems multiple GPUs will appear on the
        // same PCIe bus. A unique identifier for the GPU is the UUID,
        // which we can just print the first 8 bytes from.
        printf( "rank %d, dev %d, prop %s [%X]\n",
                rank, dev, prop.name, ((unsigned long long*)prop.uuid.bytes)[0]);
    }
}
#endif

int main(int argc, char *argv[])
{
  int myid, mydeviceid, numprocs,numdevices;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  
  Timer stopwatch;
  
#if FVSAND_HAS_GPU
  FVSAND_GPU_CHECK_ERROR(cudaGetDeviceCount(&numdevices));
  mydeviceid = myid % numdevices;
  FVSAND_GPU_CHECK_ERROR(cudaSetDevice(mydeviceid));
  listdev(myid);
  //printf( "[rank %d, cnt %d, deviceid %d]\n", myid, numdevices, mydeviceid);
#endif
  // default parameters
  char fname[64]="data.tri";
  double dsmin=0.01;
  double stretch=1.1;
  int nlevels=30;
  int nfields=5;
  std::vector<double> flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4};
  int nsteps=2000;
  int nsave=100;
  double dt=0.03;
  bool reOrderCells=false; // Re-order cells for better cache efficiency 
  int nsweep = 2;   // Jacobi Sweeps (=0 means explict)
  int istoreJac =3; // Jacobian storage or not 
  int restype=0;    // restype = 0 (cell-based) 1 (face-based)
  if (argc > 1) {
   parseInputs(argv[1],fname,&dsmin,&stretch,&nlevels,
	      flovar,&nsteps,&nsave,&dt,reOrderCells,&nsweep,
	      &istoreJac,&restype);
  }
  
  // runge-kutta tableue
  double rk[4]={0.25,8./15,5./12,3./4};

  // create strand mesh
  StrandMesh *sm;
  sm=new StrandMesh(fname,dsmin,stretch,nlevels,myid);
  if (reOrderCells) sm->ReOrderCells();
  sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);
  //sm->WriteMesh(myid);

  // create local mesh partitions
  // and compute grid metrics
  LocalMesh *lm;
  lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
  lm->CreateGridMetrics(istoreJac);

  // initialize solution
  lm->InitSolution(flovar.data(),nfields);

  stopwatch.tick();

  for(int iter=0;iter<nsteps;iter++)
    {
      std::ostringstream timestep_name;
      timestep_name << "TimeStep-" << iter;
      if(nsweep){ // implicit 
	FVSAND_NVTX_SECTION(timestep_name.str(),
         lm->Residual(lm->q,restype);           // computes res_d
   	 lm->Jacobi(lm->q,dt,nsweep,istoreJac); // runs sweeps and replaces res_d with dqtilde
         lm->UpdateQ(lm->q,lm->q,1);            // adds dqtilde (in res_d) to q XX is this dt or 1?
	);
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

     if ((iter+1)%nsave ==0 || iter==0) {
	double rnorm=lm->ResNorm();
        if (myid==0) printf("iter:%6d  %16.8e\n",iter+1,rnorm);
      }

    }
  
  double elapsed = stopwatch.tock();
  if (myid == 0) {
   printf("# ----------------------------------\n");
   printf("# Elapsed time: %13.4f s\n", elapsed);
   printf("# Through-put : %13.4f [million-elements/sec/iteration]\n",
		   sm->ncells/(elapsed/nsteps)/1e6);
   printf("# ----------------------------------\n");
  }

  // lm->WriteMesh(myid);  

  MPI_Finalize();

  return 0;
}



