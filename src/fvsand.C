#include <iostream>
#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "fvsand_gpu.h"
#include "timer.h"
#include <typeinfo>
#include <bitset>
#include <string>
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
  int myid,numprocs,numdevices;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  
  Timer stopwatch;
  
#if FVSAND_HAS_GPU
  FVSAND_GPU_CHECK_ERROR(cudaGetDeviceCount(&numdevices));
  FVSAND_GPU_CHECK_ERROR(cudaSetDevice(myid%numdevices));
  listdev(myid);
#endif

  char fname[]="data.tri";
  StrandMesh *sm;
  sm=new StrandMesh(fname,0.01,1.1,30);
  sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);
  //sm->WriteMesh(myid);

  LocalMesh *lm;
  lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
  lm->CreateGridMetrics();

  int nfields=5;
  std::vector<double> flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4};
  lm->InitSolution(flovar.data(),nfields);

  int nsteps=2000;
  int nsave=100;
  double dt=0.001;
  int restype=0;  // restype = 0 (cell-based) 1 (face-based)
  double rk[4]={0.25,8./15,5./12,3./4};

  stopwatch.tick();

  for(int iter=0;iter<nsteps;iter++)
    {

      lm->Residual(lm->q,restype);
      lm->Update(lm->qn,lm->q,rk[1]*dt);
      lm->Update(lm->q,lm->q,rk[0]*dt);

      lm->Residual(lm->qn,restype);
      lm->Update(lm->qn,lm->q,rk[2]*dt);

      lm->Residual(lm->qn,restype);
      lm->Update(lm->q,lm->q,rk[3]*dt);

      if ((iter+1)%nsave ==0) {
        double rnorm=lm->ResNorm();
        if (myid==0) printf("iter:%6d  %16.8e\n",iter+1,rnorm);
      }
    }
  
  double elapsed = stopwatch.tock();
  printf("# ----------------------------------\n");
  printf("# Elapsed time: %13.4f s\n", elapsed);
  printf("# ----------------------------------\n");

  lm->WriteMesh(myid);  

  MPI_Finalize();
}



