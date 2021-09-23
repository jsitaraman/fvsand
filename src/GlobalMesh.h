#ifndef GLOBALMESH_H
#define GLOBALMESH_H

#include <vector>
#include <stdint.h>
#include <assert.h>
//
// MeshBlock class - container and functions for generic unstructured grid partition in 3D
//
// Jay Sitaraman
namespace FVSAND {
  
class GlobalMesh
{
 public :

  uint64_t nnodes;  // < number of grid nodes 
  uint64_t ncells;  // < total number of cells
  uint64_t nfaces;  // < total number of faces
  double *x;        // < grid coordinates
  int ntypes;       // < number of different types of cells 
  int *nv;          // < number of vertices for each type of cell
  uint64_t *cell2node;  // < cell2node connectivity
  uint64_t *nc;         // < number of each of different kinds of cells
                        // < (tets, prism, pyramids, hex etc)
  int *procmap;         // < process map for each cell
  int *ncon;            // < number of connections per cell
  int64_t *cell2cell;  // < connectivity map of each cell
  int64_t *faceInfo;   // < face connectivity and face to cell map
  int *nconn;           // < number of connections per cell
  //
  // number of vertices per face for all regular polyhedra and
  // their implicit connectivity, using Dimitri's mcell style description
  // http://scientific-sims.com/oldsite/index.php/solver-file-formats/cell-type-definitions-dm
  //
  int numverts[4][6]={3,3,3,3,0,0,4,3,3,3,3,0,3,4,4,4,3,0,4,4,4,4,4,4};
  int face2node[4][24]={1,2,3,3,1,4,2,2,2,4,3,3,1,3,4,4,0,0,0,0,0,0,0,0,
			1,2,3,4,1,5,2,2,2,5,3,3,4,3,5,5,1,4,5,5,0,0,0,0,
			1,2,3,3,1,4,5,2,2,5,6,3,1,3,6,4,4,6,5,5,0,0,0,0,
			1,2,3,4,1,5,6,2,2,6,7,3,3,7,8,4,1,4,8,5,5,8,7,6};
  GlobalMesh() 
    {
      nnodes=0;
      ncells=0;
      ntypes=0;
      nv=NULL;
      nc=NULL;
    };
  virtual ~GlobalMesh() {
    delete [] x;
    delete [] nv;
    delete [] cell2node;
    delete [] nc;
    delete [] procmap;
    delete [] cell2cell;
    delete [] nconn;
  }
  
};

class StrandMesh : public GlobalMesh
{
  public:
  StrandMesh(char *surface_file,double ds, double stretch, int nlevels,int myid);
  void ReOrderCells(void);
  void WriteMesh(int label);
  void WriteBoundaries(int label);
  void PartitionSphereMesh(int, int, MPI_Comm);
  ~StrandMesh() {};
};
  
}
#endif /* GLOBALMESH_H */
