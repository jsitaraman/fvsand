#pragma once
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


class PyFV {
  Timer stopwatch;
  int myid, mydeviceid, numprocs,numdevices;
  int meshtype, nsubit;

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
  pybind11::dict get_grid_data();
};

// check big or little endian: cast as char array and check first byte
static const unsigned int tmpx  = 0x76543210;
static const bool little_endian = (((char*)&tmpx)[0] == 0x10);

class GPUArray {
public:
  pybind11::dict d;
  GPUArray(int* ptr, int size){
    d["shape"]   = Py_BuildValue("(i)", size);
    d["data"]    = Py_BuildValue("(K,O)", (unsigned long)ptr, Py_False); // read only is false
    d["version"] = 2;
    d["typestr"] = (little_endian)? "<i4" : ">i4";
  };
  GPUArray(float* ptr, int size){
    d["shape"]   = Py_BuildValue("(i)", size);
    d["data"]    = Py_BuildValue("(K,O)", (unsigned long)ptr, Py_False); // read only is false
    d["version"] = 2;
    d["typestr"] = (little_endian)? "<f4" : ">f4";
  };
  GPUArray(double* ptr, int size){
    d["shape"]   = Py_BuildValue("(i)", size);
    d["data"]    = Py_BuildValue("(K,O)", (unsigned long)ptr, Py_False); // read only is false
    d["version"] = 2;
    d["typestr"] = (little_endian)? "<f8" : ">f8";
  };
};
