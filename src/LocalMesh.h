#ifndef LOCALMESH_H
#define LOCALMESH_H

#include <vector>
#include <stdint.h>
#include <assert.h>
#include <map>
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
  //
  // host data populated by partitoner 
  //
  std::vector<double> x;        // < grid coordinates
  std::vector<int>  nvcft;      // < vertex connectivity location for each cell
  std::vector<int> cell2node;   // < cell2node connectivity
  std::vector<int> cell2cell;   // < cell2cell connectivity
  std::vector<int> ncon;        // < number of neighbors per cell
  std::vector<uint64_t> local2global;    // local2global numbering including ghost cells
  std::map<uint64_t, int > global2local;// global2local map including ghost cells 
  std::map< int, std::vector<int>> sndmap; // map of send data (procid, local id of owned cells)
  std::map <int, std::vector<int>> rcvmap; // map of recv data (procid, localid of ghost cells)
  //
  // device data
  //
  // grid metrics
  //
  double *x_d;
  int *cell2node_d,*ncon_d,*nvcft_d,*nccft_h,*nccft_d,*cell2cell_d;
  double *center;
  double *normals;
  double *volume;
  // gradient weights
  double *lsqwts;

  int nthreads,n_blocks;
  int block_size{128};
  
 public:

  // solution fields at n+1,n & n-1
  double *q,*qn,*qnn;
  
  LocalMesh() {}; 
  ~LocalMesh();
  LocalMesh(GlobalMesh *g,
	    int myid,
	    MPI_Comm comm);
  void WriteMesh(int label);
  void createGridMetrics();
  void initSolution(double *, int);
};
  
}
#endif /* LOCALMESH_H */
