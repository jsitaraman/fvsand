#include <iostream>
#include "mpi.h"
#include "GlobalMesh.h"
#include "LocalMesh.h"
#include "timer.h"
#include <typeinfo>
#include <bitset>
#include <string>
#include "NVTXMacros.h"
#include <sstream> // for std::ostringstream

class PyFV {
  Timer stopwatch;
  int myid, mydeviceid, numprocs,numdevices;

  // default params
  double dsmin,stretch,dt;
  int nlevels,nfields,nsteps,nsave,nsweep,istoreJac,restype;
  bool reOrderCells;
  std::vector<double> flovar;

  double rk[4]={0.25,8./15,5./12,3./4};

  FVSAND::StrandMesh *sm;
  FVSAND::LocalMesh *lm;

public:
  PyFV(std::string inputfile);
  ~PyFV();
  void step(int iter);
};

