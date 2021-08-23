#include <iostream>
#include "GlobalMesh.h"

int main(int argc, char *argv[])
{
  char fname[]="data.tri";
  StrandMesh *sm;
  sm=new StrandMesh(fname,0.01,1.1,20);
  sm->WriteMesh(0);
}



