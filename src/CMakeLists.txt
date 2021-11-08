PROJECT(fvsand)
ENABLE_LANGUAGE(Fortran C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

# If CMAKE_BUILD_TYPE is not specified, default to "Debug"
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
  
  message( STATUS "Setting build type to [Debug]" )
  set( CMAKE_BUILD_TYPE "Debug" CACHE STRING 
       "Choose the type of build." FORCE )

  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
                "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )

endif()

option(FVSAND_ENABLE_CUDA "Enable CUDA support (default: off)" ON)

# CUDA specific options
set(FVSAND_CUDA_SM "70" CACHE STRING "CUDA arch option")
#set(FVSAND_HIP_ARCH "gfx906,gfx908" CACHE STRING "HIP arch option")

#SET(CMAKE_C_FLAGS "-fPIC -g -rdynamic")
#SET(CMAKE_CXX_FLAGS "-fPIC -g -rdynamic -O0")
#SET(CMAKE_Fortran_FLAGS "-fbounds-check -fbacktrace -g")

SET(CMAKE_C_FLAGS "-g -fPIC -rdynamic ")
SET(CMAKE_CXX_FLAGS "-g -fPIC -rdynamic ")
SET(CMAKE_Fortran_FLAGS "-g -fdefault-real-8")

if (FVSAND_ENABLE_CUDA)
  enable_language(CUDA)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION}. "
      "Please use CUDA toolkit version 9.0 or newer")
  endif()
  set(FVSAND_CUDA_FLAGS "-rdynamic --Wno-deprecated-gpu-targets -m64")
  set(FVSAND_CUDA_ARCH_FLAGS "-gencode=arch=compute_${FVSAND_CUDA_SM},code=sm_${FVSAND_CUDA_SM}")
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_Fortran_INCLUDE_PATH})

ADD_EXECUTABLE(fvsand.exe)
target_sources(fvsand.exe PRIVATE
  # Fortran sources
  boundaryface_interface.f90
  partSphere.f90
  clust.f
  findfact.f90
  divide1D.f90
  # CXX sources
  fvsand.cpp
  StrandMesh.cpp
  CartesianMesh.cpp
  LocalMesh.C
  utils.c
  timer.cpp
  Range.C
  rcm.cpp
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} /usr/local/include/)

if (FVSAND_ENABLE_CUDA)
  separate_arguments(FVSAND_CUDA_FLAGS)
  target_compile_definitions(fvsand.exe PUBLIC
    FVSAND_HAS_GPU FVSAND_HAS_CUDA)
  target_compile_options(fvsand.exe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${FVSAND_CUDA_FLAGS}>)
  set(CMAKE_CUDA_FLAGS ${FVSAND_CUDA_ARCH_FLAGS})
  set_target_properties(fvsand.exe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
  get_target_property(FVSANDLIB_CPP_SRC fvsand.exe SOURCES)
  list(FILTER FVSANDLIB_CPP_SRC INCLUDE REGEX "\\.C")
  set_source_files_properties(${FVSANDLIB_CPP_SRC} PROPERTIES LANGUAGE CUDA)

  find_package(CUDAToolkit)
  target_link_libraries(fvsand.exe CUDA::nvToolsExt)
  
endif()

TARGET_LINK_LIBRARIES(fvsand.exe m gfortran ${MPI_LIBRARIES} )
