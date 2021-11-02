#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pyfv.h"

namespace py = pybind11;

PYBIND11_MODULE(pyfv, m) {
  m.doc() = "FVSand library wrapped with pybind11"; // module docstring


  //
  // Main class for driving FVSand from Python.
  py::class_<PyFV>(m, "PyFV")
    .def(py::init<std::string>())   // <-- we have a constructor that takes a string
    .def("step", &PyFV::step)                    // expose "step" function to python
    .def("get_grid_data", &PyFV::get_grid_data)  // ''
    ;

  //
  // This is a light wrapper class for passing CUDA data to python in
  // a way compatible with the cupy library (like numpy).
  py::class_<GPUArray>(m, "GPUArray")
    .def_readwrite("__cuda_array_interface__", &GPUArray::d);


}
