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
  for(int i=0;i<3*nnodes;i++) x_d[i]*=scale;
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
  viscousdivergence_mms_(centroid_d,res_mms_visc,&nfields_d,&N);
  Residual(q, 2, dt, 0);
  mms_error=0.0;
  std::vector<bool> use_cells(ncells,true);
  for(int i=0;i<ncells;i++) {
    for(int j=0;j<6;j++) {
      if (cell2cell[6*i+j] < 0) {
	use_cells[i]=false;
	break;
      }
      int ineig=cell2cell[6*i+j];
      for (int k=0;k<6 && use_cells[i];k++) {
	if (cell2cell[6*ineig+k] < 0) {
	  use_cells[i]=false;
	}
      }
      if (!use_cells[i]) break;
    }
  }
  int nused=0;
  for(int i=0;i<ncells;i++) if (use_cells[i]) nused++;
  int nstride=ncells+nhalo;
  int imax_cell=-1;
  int ifield=-1;
  double errmax=0.0;
  double XX=-0.4375;
  double YY=-0.4375;   
  FILE *fp=fopen("mms.dat","w");
  for(int icell=0;icell<ncells;icell++)
    {
      if (use_cells[icell]) {
        for(int j=0;j<nfields_d;j++) {
	  int i=icell+j*stride;
	  double err=fabs(res_d[i]-res_mms_inviscid[i]);
	  if (err > errmax) {
	    imax_cell=icell;
	    errmax=err;
	    ifield=j;
	  }
	  mms_error+=(res_d[i]-(res_mms_inviscid[i]+res_mms_visc[i]))*
	    (res_d[i]-(res_mms_inviscid[i]+res_mms_visc[i]));

	  if ((i%nfields_d==0 && fabs(centroid_d[icell]-XX) < 1e-3 &&
	       fabs(centroid_d[icell+ncells+nhalo]-YY) < 1e-3)) {
	    fprintf(fp,"%f %f %f %f %f %f %f %f %f %f %f %f %f\n",centroid_d[icell],centroid_d[icell+(ncells+nhalo)],
		    centroid_d[icell+2*(ncells+nhalo)],
		    res_d[icell],
		    res_d[icell+nstride],
		    res_d[icell+2*nstride],
		    res_d[icell+3*nstride],
		    res_d[icell+4*nstride],
		    res_mms_visc[icell],
		    res_mms_visc[icell+nstride],
		    res_mms_visc[icell+2*nstride],
		    res_mms_visc[icell+3*nstride],
		    res_mms_visc[icell+4*nstride]);
	  }
	}
      }
    }
#if 0
  fclose(fp);
  printf("ncells/nused=%d %d\n",ncells,nused);
  printf("imax/errmax=%d %d %f\n",imax_cell,ifield,errmax);
  printf("%f %f %f\n",centroid_d[imax_cell],
                      centroid_d[imax_cell+stride],
                      centroid_d[imax_cell+2*stride]);
#endif
  for(int i=0;i<(ncells+nhalo)*nfields_d;i++) q[i]=res_d[i];

  mms_error=sqrt(mms_error/((ncells+nhalo)*nfields_d));
}

