#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pyfv.h"

namespace py = pybind11;

PYBIND11_MODULE(pyfv, m) {
  m.doc() = "FVSand library wrapped with pybind11"; // module docstring
  py::class_<PyFV>(m, "PyFV")
    .def(py::init<std::string>())   // <-- we have a constructor that takes a string
    .def("step", &PyFV::step)
    ;
    // .def("debug",             &PybOrchard::debug)
    // .def("seed" ,             &PybOrchard::seed_pts)
    // .def("parse_inputs" ,     &PybOrchard::parse_inputs)
    // .def("get_grid_data",     &PybOrchard::get_grid_data)
    // .def("write_grid" ,       &PybOrchard::write_grid)
    // .def("write_solution",    &PybOrchard::write_sol)
    // .def("write_restart",     &PybOrchard::write_restart)
    // .def("init_flow",         &PybOrchard::init_flow)
    // .def("step",              &PybOrchard::step)
    // .def("vtx_markers",       &PybOrchard::init_vtx_markers)
    // .def("post_process_grid", &PybOrchard::post_process_grid)
    // .def("set_accelerator",   &PybOrchard::set_accelerator)
    // .def("register_mvp_cb",   &PybOrchard::register_mvp_cb)
    // .def("norm_squared",      &PybOrchard::norm_squared)
    // .def("mvp",               &PybOrchard::mvp)
    // .def("precondition",      &PybOrchard::precondition)
    // .def("vdp",               &PybOrchard::vdp)
    // .def("compute_rhs",       &PybOrchard::compute_rhs)
    // .def("start_step",        &PybOrchard::start_step)
    // .def("end_step",          &PybOrchard::end_step)
    // .def("get_rhs",           &PybOrchard::get_rhs)
    // .def("update_dt",         &PybOrchard::update_dt)
    // .def("set_grid_speed",    &PybOrchard::set_grid_speed);

}
