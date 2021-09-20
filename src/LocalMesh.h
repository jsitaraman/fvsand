#ifndef LOCALMESH_H
#define LOCALMESH_H

#include <vector>
#include <stdint.h>
#include <assert.h>
#include <map>
#include "parallelComm.h"
//
// MeshBlock class - container and functions for generic unstructured grid partition in 3D
//
// Jay Sitaraman
namespace FVSAND {

class GlobalMesh;
  
class LocalMesh
{
 private :

  int nnodes{0};       // < number of grid nodes 
  int ncells{0};       // < total number of cells
  int nhalo{0};        // < total number of halo cells
  int nfaces{0};       // < total number of faces
  int procid{0};       // < process id where this mesh belongs
  int meshtag{0};      // < meshtag
  MPI_Comm mycomm;     // < communicator for this mesh
  int myid;            // < rank in this mesh group
  int ngroup;          // < number of ranks in this group
     
  parallelComm pc;     // parallel communicator struct

  //
  // host data populated by partitoner 
  //
  std::vector<double> x;        // < grid coordinates
  std::vector<int>  nvcft;      // < vertex connectivity location for each cell
  std::vector<int> cell2node;   // < cell2node connectivity
  std::vector<int> cell2cell;   // < cell2cell connectivity
  
  std::vector<int> ncon;        // < number of neighbors per cell
  std::vector<uint64_t> local2global;    // local2global numbering including ghost cells
  std::map<uint64_t, int > global2local; // global2local map including ghost cells
  // communication maps
  std::map< int, std::vector<int>> sndmap; // map of send data (procid, local id of owned cells)
  std::map <int, std::vector<int>> rcvmap; // map of recv data (procid, localid of ghost cells)
  std::vector<int> device2host;            // indices that needs to be pushed to host   (interior)
  std::vector<int> host2device;            // indices that needs to be pushed to device (ghost)

  int *device2host_d{nullptr};               // same data on the gpu
  int *host2device_d{nullptr};               //
  double *qbuf{nullptr};                     // storage space on host to push and pull from device
  double *qbuf2{nullptr};                    // storage space on host to push and pull from device
  double *qbuf_d{nullptr};                   // storage space on device
  double *qbuf_d2{nullptr};                  // storage space on device
  //
  // solver data
  //
  int nfields_d;                          // number of fields
  double *x_d{nullptr};                      // vertex coordinates
  double *flovar_d{nullptr},*qinf_d{nullptr};   // flow variables (primitive and conservative)  
  
  int *cell2node_d{nullptr};           // cell to node connectivity on device
  int *nvcft_d{nullptr};               // cell to node cumulative frequency table
  int *nccft_h{nullptr};               // cell to cell cumulative frequency table (host)
  int *nccft_d{nullptr};               // cell to cell cumulative frequency table (device)
  int *ncon_d{nullptr};                // number of connections per cell in cell to cell (device)
  int *cell2cell_d{nullptr};           // cell to cell connectivity graph

  double *center_d{nullptr};      // cell center  (device)
  double *normals_d{nullptr};     // cell normals (device)
  double *volume_d{nullptr};      // cell volume  (device)
  double *res_d{nullptr};         // residual (host and device)

  // jacobian quantities
  double *rmatall_d{nullptr}, *Dall_d{nullptr}; 

  // face quantities
  int *cell2face_d{nullptr};
  int *facetype_d{nullptr};
  double *facenorm_d{nullptr};
  double *faceq_d{nullptr};
  double *faceflux_d{nullptr};

  // gradient weights
  double *lsqwts{nullptr};      // least square weights

  // host/device data
  int nthreads,n_blocks;
  int block_size{128};  
  int istor{0};

  MPI_Request *ireq{nullptr};
  MPI_Status *istatus{nullptr};
  
 public:

  // solution fields at n+1,n & n-1
  double *qh{nullptr}; // scratch storage space on host
  double *q{nullptr};
  double *qn{nullptr};
  double *qnn{nullptr};
  double *dq_d{nullptr};
  double *dqupdate_d{nullptr}; 	// update on device 
  int *iblank{nullptr};   // iblanking on host
  int *iblank_d{nullptr}; // iblanking on device
  
  LocalMesh() {}; 
  ~LocalMesh();
  LocalMesh(GlobalMesh *g,
	    int myid,
	    MPI_Comm comm);
  void WriteMesh(int label);
  void CreateGridMetrics();
  void CreateFaces();
  void InitSolution(double *, int);
  void Residual(double * qv, int);
  void Residual_cell(double *qv);
  void Residual_face(double *qv);
  void Jacobi(double *qv, double, int, int);
  void Update(double *qdest, double *qsrc, double fscal);
  void UpdateQ(double *qdest, double *qsrc, double fscal);
  void UpdateFringes(double *, double *);
  void UpdateFringes(double *);
  double ResNorm(void);
};
  
}
#endif /* LOCALMESH_H */
