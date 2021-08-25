PROJECT(fvsand)
ENABLE_LANGUAGE(Fortran C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)


SET(CGNS_LOCATION /usr/local/lib)
SET(CMAKE_C_FLAGS "-fPIC -g -rdynamic")
SET(CMAKE_CXX_FLAGS "-fPIC -g -rdynamic -O0")
SET(CMAKE_Fortran_FLAGS "-fbounds-check -fbacktrace -g")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_Fortran_INCLUDE_PATH})

set(DIST_SRCS
  fvsand.C
  StrandMesh.C
  LocalMesh.C
  utils.c
  partSphere.f90
  boundaryface_interface.f90
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} /usr/local/include/)
LINK_DIRECTORIES(${CGNS_LOCATION}/lib)
ADD_EXECUTABLE(fvsand.exe ${DIST_SRCS})
TARGET_LINK_LIBRARIES(fvsand.exe m ${MPI_LIBRARIES} )
