PROJECT(fvsand)
ENABLE_LANGUAGE(Fortran C)
CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
SET(CGNS_LOCATION /usr/local/lib)
SET(CMAKE_C_FLAGS "-fPIC -g -rdynamic")
SET(CMAKE_Fortran_FLAGS "-fbounds-check -fbacktrace")

set(DIST_SRCS
  fvsand.C
  StrandMesh.C
  createCommPatterns.C
  utils.c
  partSphere.f90
  boundaryface_interface.f90
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} /usr/local/include/)
LINK_DIRECTORIES(${CGNS_LOCATION}/lib)
ADD_EXECUTABLE(fvsand.exe ${DIST_SRCS})
TARGET_LINK_LIBRARIES(fvsand.exe m)
