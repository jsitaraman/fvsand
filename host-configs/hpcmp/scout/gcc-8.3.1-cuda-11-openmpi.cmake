#------------------------------------------------------------------------------
# Compiler Settings
#------------------------------------------------------------------------------
set(CMAKE_C_COMPILER /usr/bin/gcc CACHE PATH "C compiler to use" )
set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE PATH "C++ compiler to use")
set(CMAKE_Fortran_COMPILER /usr/bin/gfortran CACHE PATH "Fortran compiler to use")

#------------------------------------------------------------------------------
# CUDA 
#------------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
#set(CUDA_ARCHITECTURES "70" CACHE)

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------
set(MPI_BIN_ROOT /p/app/restricted/create/av/helios/externals/scout_tools/openmpi-4.0.5/bin CACHE PATH "MPI bin PATH")
set(MPI_C_COMPILER ${MPI_BIN_ROOT}/mpicc CACHE PATH "MPI C compiler" )
set(MPIC_CXX_COMPILER ${MPI_BIN_ROOT}/mpicxx CACHE PATH "MPI CXX compiler")
set(MPI_Fortran_COMPILER ${MPI_BIN_ROOT}/mpif90 CACHE PATH "MPI Fortran compiler")
set(MPIEXEC_EXECUTABLE ${MPI_BIN_ROOT}/mpiexec CACHE PATH "MPIEXEC executable" )


