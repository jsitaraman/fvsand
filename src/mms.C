#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "NVTXMacros.h"
#include "timer.h"
#include "fvsand_gpu.h"
#include <cstdio>
#include <math.h>
#if defined(FVSAND_HAS_GPU) && defined(FVSAND_HAS_CUDA) && !defined(FVSAND_FAKE_GPU)
// add CUDA thrust includes
#include <thrust/transform_reduce.h>
#endif
extern "C" {
  void get_qvar_(double *,double *,double *, double *);
  void set_params_mms_(double *, double *, double *, double *);
  void invisciddivergence_mms_(double *, double *, int *, int *);
  void viscousdivergence_mms_(double *, double *, int *, int *);
}

using namespace FVSAND;

void LocalMesh::mms_init(double scale)
{
  // scale the mesh
  for(int i=0;i<nnodes;i++) x_d[i]*=scale;
  RecomputeMetrics();

  double Gamma=1.4;
  double Pr=0.72;
  double Prtr=0.90;

  set_params_mms_(&Gamma,&Pr,&Prtr,&flovar_d[5]);

  for(int i=0;i<ncells+nhalo;i++) {
    double xx=centroid_d[i];
    double yy=centroid_d[i+ncells+nhalo];
    double zz=centroid_d[i+2*(ncells+nhalo)];
    double qqv[nfields_d];
    get_qvar_(qqv,&xx,&yy,&zz);
    for(int j=0;j<nfields_d;j++)
      q[i+j*(ncells+nhalo)]=qqv[j];
  }
}

void LocalMesh::mms_compute(double &mms_error, double dt)
{
  double *res_mms_inviscid;
  double *res_mms_visc;
  res_mms_inviscid=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields_d);
  res_mms_visc=gpu::allocate_on_device<double>(sizeof(double)*(ncells+nhalo)*nfields_d);
  for(int i=0;i<(ncells+nhalo)*nfields_d;i++)
    {
     res_mms_inviscid[i]=0;
     res_mms_visc[i]=0;
    }
  int N=ncells+nhalo;
  invisciddivergence_mms_(centroid_d,res_mms_inviscid,&nfields_d,&N);
  //viscousdivergence_mms_(centroid_d,res_mms_visc,&nfields_d,&N);
  Residual(q, 2, dt, 0);
  mms_error=0.0;
  for(int i=0;i<(ncells+nhalo)*nfields_d;i++)
   mms_error+=(res_d[i]-(res_mms_inviscid[i]+res_mms_visc[i]))*
	      (res_d[i]-(res_mms_inviscid[i]+res_mms_visc[i]));

  for(int i=0;i<(ncells+nhalo)*nfields_d;i++) q[i]=res_d[i];

  mms_error=sqrt(mms_error/((ncells+nhalo)*nfields_d));
}

