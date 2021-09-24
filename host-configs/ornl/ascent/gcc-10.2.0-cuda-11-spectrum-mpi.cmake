#------------------------------------------------------------------------------
# Compiler Settings
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER /sw/ascent/gcc/10.2.0/bin/gcc CACHE PATH "C compiler to use" )
set(CMAKE_CXX_COMPILER /sw/ascent/gcc/10.2.0/bin/g++ CACHE PATH "C++ compiler to use")
set(CMAKE_Fortran_COMPILER /sw/ascent/gcc/10.2.0/bin/gfortran CACHE PATH "Fortran compiler to use")

#------------------------------------------------------------------------------
# CUDA 
#------------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------
set(MPI_BIN_ROOT /autofs/nccsopen-svm1_sw/ascent/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-10.2.0/spectrum-mpi-10.3.1.2-20200121-4pokcfvq4efu6vh7gofiiszc7t7iyzqc/bin CACHE PATH "MPI bin PATH")
set(MPI_C_COMPILER ${MPI_BIN_ROOT}/mpicc CACHE PATH "MPI C compiler" )
set(MPIC_CXX_COMPILER ${MPI_BIN_ROOT}/mpicxx CACHE PATH "MPI CXX compiler")
set(MPI_Fortran_COMPILER ${MPI_BIN_ROOT}/mpif90 CACHE PATH "MPI Fortran compiler")
set(MPIEXEC_EXECUTABLE ${MPI_BIN_ROOT}/mpiexec CACHE PATH "MPIEXEC executable" )


