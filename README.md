Finite volume sand box code to be used as mini-app for hackathon.

 - Simple prismatic mesh generated from a sphere surface
 - Arbitrary paritioning for MPI
 - Inviscid fluxes
 - Explicit RK3

- verified multi-CPU execution
- verified single GPU execution


To build on scout:

```
export CREATE_HOME=/p/app/restricted/create
export MODULEPATH=${CREATE_HOME}/modulefiles:$MODULEPATH
module purge
module load BCT
module load compiler/gcc/8.3.1
module load cuda/11.1
module load cmake/3.19
module load av/helios/miniconda/3.8
module load av/helios/openmpi/4.0.5

mkdir build;
cd build;
cmake ../src;
ccmake .
swich GPU compilation ON or OFF
make
```

j.s 08/28/2021
