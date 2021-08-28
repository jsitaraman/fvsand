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
  sm=new StrandMesh(fname,0.01,1.1,30);
  sm->PartitionSphereMesh(myid,numprocs,MPI_COMM_WORLD);
  //sm->WriteMesh(myid);

  LocalMesh *lm;
  lm= new LocalMesh(sm,myid,MPI_COMM_WORLD);
  lm->CreateGridMetrics();

  int nfields=5;
  std::vector<double> flovar = { 1.0, 0.2, 0.0, 0.0, 1./1.4};
  lm->InitSolution(flovar.data(),nfields);

  int nsteps=1000;
  int nsave=10;
  double dt=0.0001;
  double rk[4]={0.25,8./15,5./12,3./4};
  
  for(int iter=0;iter<nsteps;iter++)
    {
      
      lm->Residual(lm->q);
      lm->Update(lm->qn,lm->q,rk[1]*dt);
      lm->Update(lm->q,lm->q,rk[0]*dt);

      lm->Residual(lm->qn);
      lm->Update(lm->qn,lm->q,rk[2]*dt);

      lm->Residual(lm->qn);
      lm->Update(lm->q,lm->q,rk[3]*dt);

      if (iter %nsave ==0) {
	double rnorm=lm->ResNorm();
	if (myid==0) printf("iter:%d  %lf\n",iter,rnorm);
      }
    }
  lm->WriteMesh(myid);  
  MPI_Finalize();
}



