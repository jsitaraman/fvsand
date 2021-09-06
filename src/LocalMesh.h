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

  int *device2host_d;                      // same data on the gpu
  int *host2device_d;                      //
  double *qbuf;                           // storage space on host to push and pull from device
  double *qbuf_d;                         // storage space on device
  //
  // solver data
  //
  int nfields_d;                    // number of fields
  double *x_d;                      // vertex coordinates
  double *flovar_d,*qinf_d;         // flow variables (primitive and conservative)  
  
  int *cell2node_d;                 // cell to node connectivity on device
  int *nvcft_d;                     // cell to node cumulative frequency table
  int *nccft_h;                     // cell to cell cumulative frequency table (host)
  int *nccft_d;                     // cell to cell cumulative frequency table (device)
  int *ncon_d;                      // number of connections per cell in cell to cell (device)
  int *cell2cell_d;                 // cell to cell connectivity graph

  double *center_d;      // cell center  (device)
  double *normals_d;     // cell normals (device)
  double *volume_d;      // cell volume  (device)
  double *res_d;         // residual (host and device)

  // face quantities
  int *cell2face_d;
  int *facetype_d;
  double *facenorm_d;
  double *faceq_d;
  double *faceflux_d;

  // gradient weights
  double *lsqwts;      // least square weights

  // host/device data
  int nthreads,n_blocks;
  int block_size{1024};  
  int istor{0};
  
 public:

  // solution fields at n+1,n & n-1
  double *qh; // scratch storage space on host
  double *q,*qn,*qnn;
  
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
  void Update(double *qdest, double *qsrc, double fscal);
  void UpdateFringes(double *, double *);
  double ResNorm(void);
};
  
}
#endif /* LOCALMESH_H */
