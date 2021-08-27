#include <iostream>
#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
using namespace FVSAND;

int main(int argc, char *argv[])
{
  int myid,numprocs,ierr;
  ierr=MPI_Init(&argc, &argv);
  ierr=MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  ierr=MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

  char fname[]="data.tri";
  StrandMesh *sm;
  sm=new StrandMesh(fname,0.01,1.1,20);
  sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);
  //sm->WriteMesh(myid);

  LocalMesh *lm;
  lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
  lm->createGridMetrics();

  int nfields=5;
  std::vector<double> flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4};
  lm->initSolution(flovar.data(),nfields);
  lm->WriteMesh(myid);
  
  MPI_Finalize();
}



