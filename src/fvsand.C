#include <iostream>
#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
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
        printf( "rank %d, dev %d, prop %s, pci %d, %d, %d\n",
                rank, dev,
                prop.name,
                prop.pciBusID,
                prop.pciDeviceID,
                prop.pciDomainID );
    }
}
#endif

int main(int argc, char *argv[])
{
  int myid,numprocs,numdevices;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

#if FVSAND_HAS_GPU
  cudaGetDeviceCount(&numdevices);
  cudaSetDevice(myid%numdevices);
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

  int nsteps=2;
  int nsave=1;
  double dt=0.001;
  int nsweep = 2; // Jacobi Sweeps
  int restype=0;  // restype = 0 (cell-based) 1 (face-based)
  double rk[4]={0.25,8./15,5./12,3./4};

  for(int iter=0;iter<nsteps;iter++)
    {
      if(dt){ // implicit 
        lm->Residual(lm->q,restype);
	lm->Jacobi(lm->q,dt,nsweep);
        lm->Update(lm->qn,lm->q,dt); // XX is this dt or 1? 
      } else { // explicit rk solver
        lm->Residual(lm->q,restype);
        lm->Update(lm->qn,lm->q,rk[1]*dt);
        lm->Update(lm->q,lm->q,rk[0]*dt);

        lm->Residual(lm->qn,restype);
        lm->Update(lm->qn,lm->q,rk[2]*dt);

        lm->Residual(lm->qn,restype);
        lm->Update(lm->q,lm->q,rk[3]*dt);  
      }
      if (iter %nsave ==0) {
	double rnorm=lm->ResNorm();
	if (myid==0) printf("iter:%d  %lf\n",iter,rnorm);
      }
    }
  lm->WriteMesh(myid);  
  MPI_Finalize();
}



