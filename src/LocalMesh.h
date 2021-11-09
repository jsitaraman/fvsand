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
  int nobc{0};         // < number of overset boundary nodes
  int nwbc{0};         // < number of wall boundary nodes
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
  std::vector<int> obcnode;     // < overset boundary node list
  std::vector<int> wbcnode;     // < wall boundary node list
  
  std::vector<int> ncon;        // < number of neighbors per cell
  std::vector<uint64_t> local2global;    // local2global numbering including ghost cells
  std::unordered_map<uint64_t, int > global2local; // global2local map including ghost cells
  // communication maps
  std::unordered_map< int, std::vector<int>> sndmap; // map of send data (procid, local id of owned cells)
  std::unordered_map <int, std::vector<int>> rcvmap; // map of recv data (procid, localid of ghost cells)
  std::vector<int> device2host,device2host_grad;     // indices that needs to be pushed to host   (interior)
  std::vector<int> host2device,host2device_grad;    // indices that needs to be pushed to device (ghost)

  int *device2host_d{nullptr};               // same data on the gpu
  int *host2device_d{nullptr};               //
  int *device2host_grad_d{nullptr};          // 
  int *host2device_grad_d{nullptr};          //
  int buffer_size{0};
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
  int *obcnode_d{nullptr};             // overset boundary node list on device
  int *wbcnode_d{nullptr};             // wall boundary node list on device

  double *center_d{nullptr};      // cell center  (device)
  double *centroid_d{nullptr};    // cell centroid (device)
  double *facecentroid_d{nullptr};// face centroid (device)
  double *normals_d{nullptr};     // cell normals (device)
  double *volume_d{nullptr};      // cell volume  (device)
  double *grad_d{nullptr};        // gradients (device)
  double *gradweights_d{nullptr}; // gradient weights

  // jacobian quantities
  double *rmatall_d{nullptr}, *Dall_d{nullptr}; 
  float *rmatall_d_f{nullptr},*Dall_d_f{nullptr};
  
  // face quantities
  int *cell2face_d{nullptr};
  int *face2cell_d{nullptr};
  int *facetype_d{nullptr};
  double *facenorm_d{nullptr};
  double *faceq_d{nullptr};
  double *faceflux_d{nullptr};

  // gradient weights
  double *lsqwts{nullptr};      // least square weights

  // host/device data
  int nthreads,n_blocks;
  int block_size{128};  
  int istor{1};
  double order{1.0};
  double cfl{10};
  int scale,stride;

  MPI_Request *ireq{nullptr};
  MPI_Status *istatus{nullptr};

#include "faceOrder.h"
  
 public:

  // solution fields at n+1,n & n-1
  double *qh{nullptr}; // scratch storage space on host
  double *q{nullptr};
  double *qn{nullptr};
  double *qnn{nullptr};
  double *dq_d{nullptr};
  double *dqupdate_d{nullptr}; 	// update on device 
  double *res_d{nullptr};         // residual (host and device)
  double *dqres_d{nullptr};       // linear residual (host and device)
  int *iblank{nullptr};   // iblanking on host
  int *iblank_d{nullptr}; // iblanking on device
  
  LocalMesh() {}; 
  ~LocalMesh();
  LocalMesh(GlobalMesh *g,
	    int myid,
	    MPI_Comm comm);
  void WriteMesh(int label);
  void CreateGridMetrics(int);
  void CreateFaces();
  void InitSolution(double *, int);
  void Residual(double * qv, int restype, double dt=0.0, int istoreJac=0);
  void Residual_cell(double *qv);
  void Residual_cell_2nd(double *qv);
  void Residual_face(double *qv);
  void Residual_Jacobian(double *qv, double dt);
  void Residual_Jacobian_diag(double *qv, double dt);
  void Residual_Jacobian_diag_face(double *qv, double dt);
  void Residual_Jacobian_diag_face2(double *qv, double dt);
  void Residual_Jacobian_diag_2nd(double *qv, double dt);
  void Jacobi(double *qv, double, int, int);
  void Update(double *qdest, double *qsrc, double fscal);
  void UpdateQ(double *qdest, double *qsrc, double fscal);
  void RegulateDQ(double *q);
  void UpdateFringes(double *, double *);
  void UpdateFringes(double *);
  void GetGridData(double** x_hd, int* nnode_out, int* ncell_out, 
                   int** nvcft_hd, int ** nccft_hd,
		   int** cell2node_hd, int* nc2n,
		   int **cell2cell_hd, int *nc2c,
		   int **obcnode_hd,   int *nobc_out,
		   int **wbcnode_hd,   int *nwbc_out);


  void UpdateFringes_grad(double *);
  double ResNorm(double *);
  void update_time(void);
  void add_time_source(int, double , double *, double *, double *);
  void CreateBoundaryNodeLists();
};
  
}
#endif /* LOCALMESH_H */
