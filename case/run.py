import sys
sys.path.append("/home/dylan/work/codes/fvsand/build")
import mpi4py.MPI as MPI
import pyfv

nbe = pyfv.PyFV("input.fvsand.coarse")

for i in range(200):
    nbe.step(i)
