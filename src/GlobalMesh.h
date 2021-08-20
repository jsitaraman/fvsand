#ifndef GLOBALMESH_H
#define GLOBALMESH_H

#include <vector>
#include <stdint.h>
#include <assert.h>
//
// MeshBlock class - container and functions for generic unstructured grid partition in 3D
//
// Jay Sitaraman

class GlobalMesh
{
 public :

  uint64_t nnodes;  // < number of grid nodes 
  uint64_t ncells;  // < total number of cells
  double *x;        // < grid coordinates
  int ntypes;       // < number of different types of cells 
  int *nv;          // < number of vertices for each type of cell
  uint64_t *cell2node;  // < cell2node connectivity
  uint64_t *nc;         // < number of each of different kinds of cells
                        // < (tets, prism, pyramids, hex etc)
  int *procmap;         // < process map for each cell
  uint64_t *cell2cell;  // < connectivity map of each cell
  int *nconn;           // < number of connections per node

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
  StrandMesh(char *surface_file,double ds, double stretch, int nlevels);
  void WriteMesh(int label);
  ~StrandMesh() {};
};
  

#endif /* GLOBALMESH_H */
