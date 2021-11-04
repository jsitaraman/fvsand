#include<cstdio>
#include<iostream>
#include<vector>

void parseInputs(char *inputfile,
		 char *meshfile,
		 double *dsmin,
		 double *stretch,
		 int *nlevels,
		 std::vector<double>& flovar,
		 int *nsteps,
		 int *nsave,
		 double *dt,
                 bool &reOrderCells,
		 int *nsweep,
		 int *istoreJac,
		 int *restype)
{
  FILE *fp;
  char line[256],b[8];
  //char comments[100];
  fp=fopen(inputfile,"r");
   if ( fp == nullptr ) {
    printf("Could not open file [%s]\n", inputfile );
    MPI_Abort( MPI_COMM_WORLD, -1 );
  }
  fgets(line,256,fp);  sscanf(line,"meshfile=%s",meshfile);
  fgets(line,256,fp);  sscanf(line,"dsmin=%lf",dsmin);
  fgets(line,256,fp);  sscanf(line,"stretch=%lf",stretch);
  fgets(line,256,fp);  sscanf(line,"nlevels=%d",nlevels);
  fgets(line,256,fp);  sscanf(line,"density=%lf",&flovar[0]);
  fgets(line,256,fp);  sscanf(line,"Vx=%lf",&flovar[1]);
  fgets(line,256,fp);  sscanf(line,"Vy=%lf",&flovar[2]);
  fgets(line,256,fp);  sscanf(line,"Vz=%lf",&flovar[3]);
  fgets(line,256,fp);  sscanf(line,"pressure=%lf",&flovar[4]);
  fgets(line,256,fp);  sscanf(line,"nsteps=%d",nsteps);
  fgets(line,256,fp);  sscanf(line,"nsave=%d",nsave);
  fgets(line,256,fp);  sscanf(line,"dt=%lf",dt);
  fgets(line,256,fp);  sscanf(line,"nsweep=%d",nsweep);
  fgets(line,256,fp);  sscanf(line,"istoreJac=%d",istoreJac);
  fgets(line,256,fp);  sscanf(line,"restype=%d",restype);
  fgets(line,256,fp);  
  if ( sscanf(line,"reOrderCells=%[TtRrUuEe]",b))
    reOrderCells = true;
  else
    reOrderCells = false;
  fgets(line,256,fp);  sscanf(line,"rey=%lf",&flovar[5]);
  fclose(fp);
}

// int main(void)
// {
//   char inputfile[]="input.fvsand";
//   char meshfile[40];
//   double dsmin,stretch,dt;
//   int nlevels,nsteps,nsave,nsweep,istoreJac,restype;
//   std::vector<double>flovar(5,0);
  
//   parseInputs(inputfile,meshfile,&dsmin,&stretch,&nlevels,flovar,
// 	      &nsteps,&nsave,&dt,&nsweep,&istoreJac,&restype);
//   printf("%s\n",meshfile);
//   printf("%lf %lf %d\n",dsmin,stretch,nlevels);
//   printf("%lf %lf %lf %lf %lf\n",flovar[0],flovar[1],flovar[2],flovar[3],flovar[4]);
//   printf("%d %d %lf\n",nsteps,nsave,dt);
//   printf("%d %d %d\n",nsweep,istoreJac,restype);
// }
