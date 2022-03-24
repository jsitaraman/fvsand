from numpy import *
import os
tmp=open('cart.template','r').read()
idim=17
for p in range(10):
  c=tmp.replace("<idim>","%d"%idim)
  open('cart.dat','w').write(c)
  a=[b for b in os.popen('../build/fvsand.exe input.fvsand.coarse').readlines()]
  err=float(a[0].split()[1])
  dx=4./(idim-1)
  if (p > 0) :
     print(dx,err,(log(errold)-log(err))/(log(dxold)-log(dx)))
  else:
     print(dx,err)
  dxold=dx
  errold=err
  idim+=4
