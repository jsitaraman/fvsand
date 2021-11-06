#define Gam 1.4
#define Rgas (1.0/Gam)
#define c2b 0.3678
#define pr 0.72
#define prtr 0.90
#define VEPS 1e-7
FVSAND_GPU_DEVICE
void viscous_flux_fp( real & f1, real & f2, real & f3, real & f4, real & f5,
		      const real & rlft, const real& ulft, const real & vlft, const real & wlft, real & plft,
		      const real & rrht, const real& urht, const real & vrht, const real & wrht, real & prht,
		      real *grad,
		      const real & cx, const real &cy, const real & cz,
		      const real & dx, const real &dy, const real & dz,
		      const real  & rey,
		      const real & nx, const real & ny, const real & nz)
{
  real Tx,Ty,Tz;
  real rx,ry,rz;
  real tx,ty,tz;
  real nmag2,delta,scal,dot,rmag,divu,kcond;
  real mu,T,u,v,w,rho,p;
  enum viscousComponents {rho_x,rho_y,rho_z,u_x,u_y,u_z,v_x,v_y,v_z,
			  w_x,w_y,w_z,p_x,p_y,p_z};
  //for(int i=0;i<15;i++) grad[i]=0;
  // Average temperature gradient
  p=0.5*(plft+prht);
  rho=0.5*(rlft+rrht);
  Tx=(1/Rgas)*(grad[p_x]/rho - p*grad[rho_x]/(rho*rho));
  Ty=(1/Rgas)*(grad[p_y]/rho - p*grad[rho_y]/(rho*rho));
  Tz=(1/Rgas)*(grad[p_z]/rho - p*grad[rho_z]/(rho*rho));
  // vector connecting centroids
  rx=dx-cx;
  ry=dy-cy;
  rz=dz-cz;
  // tangential component of the centroid vector
  nmag2=nx*nx+ny*ny+nz*nz;
  dot=rx*nx+ry*ny*rz*nz;
  tx=rx - (dot)*nx/nmag2;
  ty=ry - (dot)*ny/nmag2;
  tz=rz - (dot)*nz/nmag2;
  //
  rmag=sqrt(rx*rx+ry*ry+rz*rz);
  rx/=rmag;
  ry/=rmag;
  rz/=rmag;
  // Temperature gradient with corrections
  // replace the component along the centroid vector
  // with direct difference.
  // scale the average gradient to satisfy criteria that
  // the projection along tangent vector is equal to the
  // direct difference
  delta=(1./(Rgas))*(prht/rrht - plft/rlft);
  scal=fvsand_min(1.0,fabs(delta/(Tx*tx+Ty*ty+Tz*tx+VEPS)));
  Tx*=scal;
  Ty*=scal;
  Tz*=scal;
  dot=(Tx*rx+Ty*ry+Tz*rz);
  Tx-=(dot*rx-delta*rx/rmag);
  Ty-=(dot*ry-delta*ry/rmag);
  Tz-=(dot*rz-delta*rz/rmag);
  //
  // velocity gradient with corrections
  // u-component
  delta=(urht-ulft);
  scal=fvsand_min(1.0,fabs(delta/(grad[u_x]*tx+grad[u_y]*ty+grad[u_z]*tz+VEPS)));
  grad[u_x]*=scal;
  grad[u_y]*=scal;
  grad[u_z]*=scal;
  dot=grad[u_x]*rx+grad[u_y]*ry+grad[u_z]*rz;
  grad[u_x]-=(dot*rx-delta*rx/rmag);
  grad[u_y]-=(dot*ry-delta*ry/rmag);
  grad[u_z]-=(dot*rz-delta*rz/rmag);
  // v-component
  delta=(vrht-vlft);
  scal=fvsand_min(1.0,fabs(delta/(grad[v_x]*tx+grad[v_y]*ty+grad[v_z]*tz+VEPS)));
  grad[v_x]*=scal;
  grad[v_y]*=scal;
  grad[v_z]*=scal;
  dot=grad[v_x]*rx+grad[v_y]*ry+grad[v_z]*rz;
  grad[v_x]-=(dot*rx-delta*rx/rmag);
  grad[v_y]-=(dot*ry-delta*ry/rmag);
  grad[v_z]-=(dot*rz-delta*rz/rmag);
  // w-component
  delta=(wrht-wlft);
  scal=fvsand_min(1.0,fabs(delta/(grad[w_x]*tx+grad[w_y]*ty+grad[w_z]*tz+VEPS)));
  grad[w_x]*=scal;
  grad[w_y]*=scal;
  grad[w_z]*=scal;
  dot=grad[w_x]*rx+grad[w_y]*ry+grad[w_z]*rz;
  grad[w_x]-=(dot*rx-delta*rx/rmag);
  grad[w_y]-=(dot*ry-delta*ry/rmag);
  grad[w_z]-=(dot*rz-delta*rz/rmag);
  // sutherlands law for laminar viscosity
  T=0.5*(plft/rlft+prht/rrht)*(1.0/Rgas);
  mu=(c2b+1)*T*sqrt(T)/(c2b+T);
  mu/=rey;
  // divergence of velocity
  divu=grad[u_x]+grad[v_y]+grad[w_z];
  // momentum viscous terms
  // tau=mu*((gradu+gradu^T)-2/3*Div u)
  f2-=(mu*((2*grad[u_x]-2.0/3.0*divu)*nx +
	   (  grad[u_y]+grad[v_x]   )*ny +
	   (  grad[u_z]+grad[w_x]   )*nz));
  f3-=(mu*((  grad[v_x]+grad[u_y]   )*nx +
	   (2*grad[v_y]-2.0/3.0*divu)*ny +
	   (  grad[v_z]+grad[w_y]   )*nz));
  f4-=(mu*((  grad[w_x]+grad[u_z]   )*nx +
	   (  grad[w_y]+grad[v_z]   )*ny +
	   (2*grad[w_z]-2.0/3.0*divu)*nz));
  // heat conductivity
  kcond=(1.0/((Gam-1)*pr))*mu;
  u=(ulft+urht)*0.5;
  v=(vlft+vrht)*0.5;
  w=(wlft+wrht)*0.5;
  f5-=(mu*((2*grad[u_x]-2.0/3.0*divu)*u+(grad[v_x]+grad[u_y])*v+(grad[w_x]+grad[u_z])*w)*nx+kcond*Tx*nx);
  f5-=(mu*((grad[u_y]+grad[v_x])*u+(2*grad[v_y]-2.0/3.0*divu)*v+(grad[w_y]+grad[v_z])*w)*ny+kcond*Ty*ny);
  f5-=(mu*((grad[u_z]+grad[w_x])*u+(grad[v_z]+grad[w_y])*v+(2*grad[w_z]-2.0/3.0*divu)*w)*nz+kcond*Tz*nz);
}
// viscous flux and Jacobian
// without cross terms. Does not need explicit gradients
FVSAND_GPU_DEVICE
void viscousFluxAndJacobian(double *faceNormal, const double &faceSpeed,
			    double *ql, double *qr,
			    double *flux,
			    double dfluxl[6][6], double dfluxr[6][6],
			    const double &rey)
{
  double ur[3],ul[3];
  int n,m;
  double gm1;
  double rj1,rj2,rj3;
  double a1,a2,a3,a4,a5,a6,a7;
  double dre,sigma,rgasi,gm1Pr;
  double t1,mu,t2,mut,fac,rr;
  double rhoi,rhoi1,rhoi2,rhoi12;
  double ux,vx,wx,uux,vvx,wwx,uvx,vwx,wux,tx;
  double plft,prht;

  double ulmag,urmag;
  double dmul[5];
  double dmur[5];
  double dmutl,dmutr;
  double dplft[5],dprht[5];
  double dt1l[5],dt2r[5];
  double dull[3][5],durr[3][5];
  double duxl[5],duxr[5];
  double dvxl[5],dvxr[5];
  double dwxl[5],dwxr[5];
  double duuxl[5],duuxr[5];
  double dvvxl[5],dvvxr[5];
  double dwwxl[5],dwwxr[5];
  double duvxl[5],duvxr[5];
  double dvwxl[5],dvwxr[5];
  double dwuxl[5],dwuxr[5];
  double dtxl[5],dtxr[5];
  static double third=1e0/3e0;
  //
  gm1=Gam-1;
  dre=1/rey;
  sigma=pr/prtr;
  rgasi=1/Rgas;
  gm1Pr=1/gm1/pr;
  //
  rhoi=1/ql[0];
  rhoi1=1/qr[0];
  rhoi2=rhoi*rhoi;
  rhoi12=rhoi1*rhoi1;
  //
  ul[0]=ql[1]/ql[0];
  ul[1]=ql[2]/ql[0];
  ul[2]=ql[3]/ql[0];
  ulmag=ul[0]*ul[0]+ul[1]*ul[1]+ul[2]*ul[2];
  //
  ur[0]=qr[1]/qr[0];
  ur[1]=qr[2]/qr[0];
  ur[2]=qr[3]/qr[0];
  urmag=ur[0]*ur[0]+ur[1]*ur[1]+ur[2]*ur[2];
  // velocity derivatives
  dull[0][0]=-ul[0]*rhoi;
  dull[0][1]=rhoi;
  dull[0][2]=dull[0][3]=dull[0][4]=0;
  
  dull[1][0]=-ul[1]*rhoi;
  dull[1][1]=0;
  dull[1][2]=rhoi;
  dull[1][3]=dull[1][4]=0;

  dull[2][0]=-ul[2]*rhoi;
  dull[2][1]=dull[2][2]=0;
  dull[2][3]=rhoi;
  dull[2][4]=0;

  durr[0][0]=-ur[0]*rhoi1;
  durr[0][1]=rhoi1;
  durr[0][2]=durr[0][3]=durr[0][4]=0;
  
  durr[1][0]=-ur[1]*rhoi1;
  durr[1][1]=0;
  durr[1][2]=rhoi1;
  durr[1][3]=durr[1][4]=0;

  durr[2][0]=-ur[2]*rhoi1;
  durr[2][1]=durr[2][2]=0;
  durr[2][3]=rhoi1;
  durr[2][4]=0;

  // pressure left and its derivative
  plft=gm1*(ql[4]-0.5*ql[0]*ulmag);
  dplft[0]=0.5*gm1*ulmag;
  dplft[1]=-gm1*ul[0];
  dplft[2]=-gm1*ul[1];
  dplft[3]=-gm1*ul[2];
  dplft[4]=gm1;
  
  t1=rgasi*plft*rhoi;
  for(n=0;n<5;n++) dt1l[n]=rgasi*dplft[n]*rhoi;
  dt1l[0]=dt1l[0]-rgasi*plft*rhoi2;
  mu=(c2b+1.)*t1*sqrt(t1)/(c2b+t1);
  for(n=0;n<5;n++) {
    dmul[n]=0.5*((c2b+1)*1.5*dt1l[n]*sqrt(t1)/(c2b+t1)+
		 -(c2b+1)*t1*sqrt(t1)/((c2b+t1)*(c2b+t1))*dt1l[n]);
  }

  prht=gm1*(qr[4]-0.5*qr[0]*urmag);  
  dprht[0]=0.5*gm1*urmag;
  dprht[1]=-gm1*ur[0];
  dprht[2]=-gm1*ur[1];
  dprht[3]=-gm1*ur[2];
  dprht[4]=gm1;
  
  t2=rgasi*prht*rhoi1;
  for(n=0;n<5;n++) dt2r[n]=rgasi*dprht[n]*rhoi1;
  dt2r[0]=dt2r[0]-rgasi*prht*rhoi12;
  mu=(c2b+1.)*t2*sqrt(t2)/(c2b+t2);
  for(n=0;n<5;n++) {
    dmur[n]=0.5*((c2b+1)*1.5*dt2r[n]*sqrt(t2)/(c2b+t2)+
		 -(c2b+1)*t2*sqrt(t2)/((c2b+t2)*(c2b+t2))*dt2r[n]);
  }

  mut=0.5*(ql[5]+qr[5]);
  dmutl=0.5;
  dmutr=0.5;
  rj1=faceNormal[0];
  rj2=faceNormal[1];
  rj3=faceNormal[2];
  a4=rj1*rj1+rj2*rj2+rj3*rj3;
  rr=sqrt(a4);
  a1=a4+third*rj1*rj1;
  a2=a4+third*rj2*rj2;
  a3=a4+third*rj3*rj3;
  a5=third*rj1*rj2;
  a6=third*rj2*rj3;
  a7=third*rj3*rj1;

  ux=ur[0]-ul[0];
  vx=ur[1]-ul[1];
  wx=ur[2]-ul[2];
  for(n=0;n<5;n++) {
    duxl[n]=-dull[0][n];
    duxr[n]=durr[0][n];
    
    dvxl[n]=-dull[1][n];
    dvxr[n]=durr[1][n];
    
    dwxl[n]=-dull[2][n];
    dwxr[n]=durr[2][n];
  }
  uux=ux*(ur[0]+ul[0]);
  vvx=vx*(ur[1]+ul[1]);
  wwx=wx*(ur[2]+ul[2]);
  uvx=(ur[0]*ur[1]-ul[0]*ul[1]);
  vwx=(ur[1]*ur[2]-ul[1]*ul[2]);
  wux=(ur[2]*ur[0]-ul[2]*ul[0]);

  for(n=0;n<5;n++) {
       
   duuxl[n]=duxl[n]*(ur[0]+ul[0])+ux*dull[0][n];
   duuxr[n]=duxr[n]*(ur[0]+ul[0])+ux*durr[0][n];
   
   dvvxl[n]=dvxl[n]*(ur[1]+ul[1])+vx*dull[1][n];
   dvvxr[n]=dvxr[n]*(ur[1]+ul[1])+vx*durr[1][n];
   
   dwwxl[n]=dwxl[n]*(ur[2]+ul[2])+wx*dull[2][n];
   dwwxr[n]=dwxr[n]*(ur[2]+ul[2])+wx*durr[2][n];
   
   duvxl[n]=-dull[0][n]*ul[1]-ul[0]*dull[1][n];
   
   duvxr[n]=durr[0][n]*ur[1]+ur[0]*durr[1][n];
   
   dvwxl[n]= -dull[1][n]*ul[2]-ul[1]*dull[2][n];            
   
   dvwxr[n]=durr[1][n]*ur[2]+ur[1]*durr[2][n];
   
   dwuxl[n]=-dull[2][n]*ul[0]-ul[2]*dull[0][n];            
   
   dwuxr[n]=durr[2][n]*ur[0]+ur[2]*durr[0][n]; 
   
  }

  tx = (t2-t1)*gm1Pr;
  for(n=0;n<5;n++) {
    dtxl[n]=-dt1l[n]*gm1Pr;
    dtxr[n]=dt2r[n]*gm1Pr;
  }
  // no density equation contribution
  flux[0]=0;
  for(n=0;n<6;n++) {
    dfluxl[0][n]=0;
    dfluxr[0][n]=0;
  }
  // momentum flux and it's jacobian
  flux[1] = a1*ux+a5*vx+a7*wx;
  flux[2] = a5*ux+a2*vx+a6*wx;
  flux[3] = a7*ux+a6*vx+a3*wx;
  for(n=0;n<5;n++) {
    dfluxl[1][n]=a1*duxl[n]+a5*dvxl[n]+a7*dwxl[n];
    dfluxl[2][n]=a5*duxl[n]+a2*dvxl[n]+a6*dwxl[n];
    dfluxl[3][n]=a7*duxl[n]+a6*dvxl[n]+a3*dwxl[n];

    dfluxr[1][n]=a1*duxr[n]+a5*dvxr[n]+a7*dwxr[n];
    dfluxr[2][n]=a5*duxr[n]+a2*dvxr[n]+a6*dwxr[n];
    dfluxr[3][n]=a7*duxr[n]+a6*dvxr[n]+a3*dwxr[n];
  }
  fac=dre;
  for(m=1;m<4;m++) {    
    for(n=0;n<5;n++) {
      dfluxl[m][n]=fac*((mu+mut)*dfluxl[m][n]+dmul[n]*flux[m]);
      dfluxr[m][n]=fac*((mu+mut)*dfluxr[m][n]+dmur[n]*flux[m]);
    }
    n=5;
    dfluxl[m][n]=fac*dmutl*flux[m];
    dfluxr[m][n]=fac*dmutr*flux[m];
  }
  for(m=1;m<4;m++) flux[m]= (mu+mut)*fac*flux[m];
  // energy equation and it's jacobian
  for(n=0;n<5;n++) {
   dfluxl[4][n]=0.5*a1*duuxl[n]+0.5*a2*dvvxl[n]+0.5*a3*dwwxl[n]+
                a5*duvxl[n] + a6*dvwxl[n] + a7*dwuxl[n]+ 
                a4*dtxl[n];
   dfluxr[4][n]=0.5*a1*duuxr[n]+0.5*a2*dvvxr[n]+0.5*a3*dwwxr[n]+
                a5*duvxr[n] + a6*dvwxr[n] + a7*dwuxr[n]+ 
                a4*dtxr[n];
  }
  flux[4] = 0.5*a1*uux + 0.5*a2*vvx + 0.5*a3*wwx +
                a5*uvx +     a6*vwx +     a7*wux + 
                a4*tx;
  for(n=0;n<5;n++) {
    dfluxl[4][n]=fac*(dmul[n]*flux[4]+(mu+mut*sigma)*dfluxl[4][n]);
    dfluxr[4][n]=fac*(dmur[n]*flux[4]+(mu+mut*sigma)*dfluxr[4][n]);
  }
  n=5;
  dfluxl[4][n]=dmutl*sigma*fac*flux[4];
  dfluxr[4][n]=dmutr*sigma*fac*flux[4];
  flux[4]=(mu+mut*sigma)*fac*flux[4];
  for(n=0;n<6;n++) {
    dfluxl[5][n]=0;
    dfluxr[5][n]=0;
  }    
  flux[5]=0;
}
