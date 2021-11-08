import sys
sys.path.append("/p/home/jaina/code/fvsand/build_python")
import mpi4py.MPI as MPI
import pyfv

nbe = pyfv.PyFV("input.fvsand.coarse")
nbe2 = pyfv.PyFV("input.fvsand.cart")

for i in range(200):
    nbe.step(i)
    nbe2.step(i)
