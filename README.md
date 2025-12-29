Finite volume sand box code to be used as mini-app for hackathon.

 - Simple prismatic mesh generated from a sphere surface
 - Arbitrary paritioning for MPI
 - Inviscid fluxes
 - Explicit RK3

- verified multi-CPU execution
- verified single GPU execution


To build:

```
cp host-configs/generic/gcc_cuda.cmake ./my_config.cmake
edit my_config.cmake and provide paths to gnu compilers and mpi
we assume nvcc is in your path

mkdir build;
cd build;
cmake -C ../my_config.cmake ../src
# optional
ccmake .
swich GPU compilation ON or OFF
make -j
```

To test:
cd case
mpiexec -n 1 ../build/fvsand.exe input.fvsand.coarse
mpiexec -n 4 ../build/fvsand.exe input.fvsand.coarse

There are medium and fine grids provided as well

j.s 08/28/2021
j.s 12/28/2025
