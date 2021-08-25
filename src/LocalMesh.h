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
namespace fvSand {

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
  // number of vertices per face for all regular polyhedra and
  // their implicit connectivity, using Dimitri's mcell style description
  // http://scientific-sims.com/oldsite/index.php/solver-file-formats/cell-type-definitions-dm
  //
  // TODO
  // This is repeated from GlobalMesh (find a common container in implementation) for now
  //
  int numverts[4][6]={3,3,3,3,0,0,4,3,3,3,3,0,3,4,4,4,3,0,4,4,4,4,4,4};
  int face2node[4][24]={1,2,3,3,1,4,2,2,2,4,3,3,1,3,4,4,0,0,0,0,0,0,0,0,
			1,2,3,4,1,5,2,2,2,5,3,3,4,3,5,5,1,4,5,5,0,0,0,0,
			1,2,3,3,1,4,5,2,2,5,6,3,1,3,6,4,4,6,5,5,0,0,0,0,
			1,2,3,4,1,5,6,2,2,6,7,3,3,7,8,4,1,4,8,5,5,8,7,6};

 public:
  
  LocalMesh() {}; 
  ~LocalMesh();
  LocalMesh(GlobalMesh *g,
	    int myid,
	    MPI_Comm comm);
  void WriteMesh(int label);
  
};
  
}
#endif /* LOCALMESH_H */
