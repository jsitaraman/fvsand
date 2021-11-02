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

option(FVSAND_ENABLE_CUDA "Enable CUDA support (default: on)" ON)

# CUDA specific options
set(FVSAND_CUDA_SM 60 70 CACHE STRING "CUDA arch option")
#set(FVSAND_HIP_ARCH "gfx906,gfx908" CACHE STRING "HIP arch option")

#SET(CMAKE_C_FLAGS "-fPIC -g -rdynamic")
#SET(CMAKE_CXX_FLAGS "-fPIC -g -rdynamic -O0")
#SET(CMAKE_Fortran_FLAGS "-fbounds-check -fbacktrace -g")

SET(CMAKE_C_FLAGS "-fPIC -rdynamic -O2")
SET(CMAKE_CXX_FLAGS "-fPIC -rdynamic -O2")
SET(CMAKE_Fortran_FLAGS "-O2")

if (FVSAND_ENABLE_CUDA)
  enable_language(CUDA)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION}. "
      "Please use CUDA toolkit version 9.0 or newer")
  endif()
  set(FVSAND_CUDA_FLAGS "-lineinfo -rdynamic --Wno-deprecated-gpu-targets -m64")
  foreach(SMCC IN LISTS FVSAND_CUDA_SM)
    # set(FVSAND_CUDA_ARCH_FLAGS "-gencode=arch=compute_${FVSAND_CUDA_SM},code=sm_${FVSAND_CUDA_SM}")
    set(FVSAND_CUDA_ARCH_FLAGS "${FVSAND_CUDA_ARCH_FLAGS} -gencode=arch=compute_${SMCC},code=sm_${SMCC}")
    message(STATUS "-gencode=arch=compute_${SMCC},code=sm_${SMCC}")
  endforeach()
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_Fortran_INCLUDE_PATH})

# -------------------------------------------------------
# Common library of FVSAND routines
#
set(sources
  # Fortran sources
  src/boundaryface_interface.f90
  src/partSphere.f90
  # CXX sources
  # src/fvsand.cpp
  src/StrandMesh.cpp
  src/LocalMesh.C
  src/utils.c
  src/timer.cpp
  src/Range.C
  src/rcm.cpp
)

add_library(fvsand_common SHARED ${sources})
if (FVSAND_ENABLE_CUDA)
  separate_arguments(FVSAND_CUDA_FLAGS)
  target_compile_definitions(fvsand_common PUBLIC FVSAND_HAS_GPU FVSAND_HAS_CUDA)
  target_compile_options(fvsand_common PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FVSAND_CUDA_FLAGS}>)
  set(CMAKE_CUDA_FLAGS ${FVSAND_CUDA_ARCH_FLAGS})
  set_target_properties(fvsand_common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  get_target_property(FVSANDLIB_CPP_SRC fvsand_common SOURCES)
  list(FILTER FVSANDLIB_CPP_SRC INCLUDE REGEX "\\.C")
  set_source_files_properties(${FVSANDLIB_CPP_SRC} PROPERTIES LANGUAGE CUDA)
  find_package(CUDAToolkit)
  target_link_libraries(fvsand_common CUDA::nvToolsExt)
endif()
target_link_libraries(fvsand_common m gfortran ${MPI_LIBRARIES} )


# -------------------------------------------------------
# Driver executable
#
add_executable(fvsand.exe "src/fvsand.cpp")
target_link_libraries(fvsand.exe fvsand_common)
if (FVSAND_ENABLE_CUDA)
  target_include_directories(fvsand.exe PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(fvsand.exe ${CUDART_LIBRARY})
endif()


# ------------------------------------------------------------------
# Python-wrapped Library
#
file(GLOB pysrc "${CMAKE_SOURCE_DIR}/interface/*.cpp")

if(NOT COMMAND pybind11_add_module)
  message(STATUS "[FVSand] WARNING: building with packaged pybind11")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/interface/pybind11")
endif()

pybind11_add_module(pyfv ${pysrc})
target_include_directories(pyfv PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pyfv PUBLIC ${CUDART_LIBRARY})
target_link_libraries(pyfv PRIVATE fvsand_common)
