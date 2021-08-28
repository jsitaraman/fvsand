#include<math.h>
#define real double
#define FOR2(i,n1,j,n2) for(int i = 0 ; i < n1 ; i++) for(int j = 0 ; j < n2 ; j++)
#define roe_max(a,b) (a>b)?a:b
FVSAND_GPU_DEVICE
void InterfaceFlux_Inviscid( real & f1, real & f2, real & f3, real & f4, real & f5,
			     real & ql1, real& ql2, real & ql3, real & ql4, real & ql5,
			     real & qr1, real& qr2, real & qr3, real & qr4, real & qr5,
			     real & dsx, real & dsy, real & dsz,
			     real & gx, real & gy, real & gz, real &spec, int & faceID)
{
  real  gm1=0.4;
  real eps,rlft,ulft,vlft,wlft,plft;
  real rlfti,rulft,rvlft,rwlft,uvwl,elft,hlft,clft;
  real rrht,urht,vrht,wrht,prht;
  real rrhti,rurht,rvrht,rwrht,uvwr,erht,hrht,crht;
  real rat,rati,rav,uav,vav,wav,hav,uvw,cav;
  real aq1,aq2,aq3,aq4,aq5,ri1,ri2,ri3,ri4,rr2,rr,r0,r1,r2,r3,r4;
  real uu,c2,c2i,auu,aupc,aumc,uulft,uurht,upclft,upcrht;
  real umclft,umcrht,dauu,dauus,daupc,daumc,daumcs,rcav,aquu;
  real daupcs,c2ih,ruuav,b1,b2,b3,b4,b5,b6,b7,b8,aj;
  real plar,eplft,eprht;

  real faceSpeed;
  spec = 0.0;

  if ( faceID == -2 ) // boundary face
    {
      plft = gm1*(ql5 - 0.5*(ql2*ql2 + ql3*ql3 + ql4*ql4)/ql1);
      f1 = 0.0;
      f2 = plft*dsx;
      f3 = plft*dsy;
      f4 = plft*dsz;
      f5 = 0.0;
      spec=0.0;
    }
  else 
    {
      //
      // first executable statement
      //
      eps = 1.e-6;
 
      faceSpeed = (gx*dsx + gy*dsy + gz*dsz)/sqrt(dsx*dsx + dsy*dsy + dsz*dsz);

      rlft = ql1;
      ulft = ql2/ql1;
      vlft = ql3/ql1;
      wlft = ql4/ql1;
      plft = gm1*(ql5 - 0.5*(ql2*ql2 + ql3*ql3 + ql4*ql4)/ql1);

      rlfti = 1.0/rlft;
      rulft = ql2;
      rvlft = ql3;
      rwlft = ql4;
      uvwl = 0.5*( ulft*ulft + vlft*vlft + wlft*wlft );
      elft = plft/gm1 + rlft*uvwl;
      hlft = ( elft + plft )*rlfti;
      clft = sqrt( gm1*( hlft - uvwl ) );

      rrht = qr1;
      urht = qr2/qr1;
      vrht = qr3/qr1;
      wrht = qr4/qr1;
      prht = gm1*(qr5-0.5*(qr2*qr2 + qr3*qr3 + qr4*qr4)/qr1);

      rrhti = 1.0/rrht;
      rurht = qr2;
      rvrht = qr3;
      rwrht = qr4;
      uvwr = 0.5*( urht*urht + vrht*vrht + wrht*wrht );
      erht = prht/gm1 + rrht*uvwr;
      hrht = ( erht + prht )*rrhti;
      crht = sqrt( gm1*( hrht - uvwr ) );

      rat  = sqrt( rrht*rlfti );
      rati = 1.0/( rat + 1. );
      rav  =   rat*rlft;
      uav  = ( rat*urht + ulft )*rati;
      vav  = ( rat*vrht + vlft )*rati;
      wav  = ( rat*wrht + wlft )*rati;
      hav  = ( rat*hrht + hlft )*rati;
      uvw  = 0.5*( uav*uav + vav*vav + wav*wav );
      cav  = sqrt( gm1*( hav - uvw ) );

      aq1  = rrht - rlft;
      aq2  = urht - ulft;
      aq3  = vrht - vlft;
      aq4  = wrht - wlft;
      aq5  = prht - plft;

      ri1 = dsx;
      ri2 = dsy;
      ri3 = dsz;
      ri4 = faceSpeed;
      rr2 = ri1*ri1 + ri2*ri2 + ri3*ri3;
      rr  = sqrt( rr2 );
      r0  = 1.0 / rr;
      r1  = ri1*r0;
      r2  = ri2*r0;
      r3  = ri3*r0;
      r4  = ri4*r0;

      uu  = r1*uav + r2*vav + r3*wav + r4;
      c2  = cav*cav;
      c2i = 1.0/c2;

      auu   = abs( uu    );
      aupc  = abs( uu+cav );
      aumc  = abs( uu-cav );

      uulft = r1*ulft + r2*vlft + r3*wlft + r4;
      uurht = r1*urht + r2*vrht + r3*wrht + r4;
      upclft= uulft + clft;
      upcrht= uurht + crht;
      umclft= uulft - clft;
      umcrht= uurht - crht;

      dauu = 4.*(uurht-uulft)+eps;
      dauus = roe_max(dauu,0.0);
      if (auu <= 0.5*dauus) 
	auu = auu*auu/dauu+0.25*dauu;
      //
      daupc = 4.*(upcrht-upclft)+eps;
      daupcs = roe_max(daupc,0.0);
  
      if (aupc <= 0.5*daupcs)  
	aupc = aupc*aupc/daupc+0.25*daupc;
      //
      daumc = 4.*(umcrht-umclft)+eps;
      daumcs = roe_max(daumc,0.0);
  
      if (aumc <= 0.5*daumcs) 
	aumc = aumc*aumc/daumc+0.25*daumc;
      //  
      spec=roe_max(auu,aupc);
      spec=roe_max(spec,aumc);
      spec=spec*rr;

      rcav = rav*cav;
      aquu = uurht - uulft;
      c2ih = 0.5*c2i;
      ruuav= auu*rav;
      b1   = auu*( aq1 - c2i*aq5 );
      b2   = c2ih*aupc*( aq5 + rcav*aquu );
      b3   = c2ih*aumc*( aq5 - rcav*aquu );
      b4   = b1 + b2 + b3;
      b5   = cav*( b2 - b3 );
      b6   = ruuav*( aq2 - r1*aquu );
      b7   = ruuav*( aq3 - r2*aquu );
      b8   = ruuav*( aq4 - r3*aquu );

      aq1 = b4;
      aq2 = uav*b4 + r1*b5 + b6;
      aq3 = vav*b4 + r2*b5 + b7;
      aq4 = wav*b4 + r3*b5 + b8;
      aq5 = hav*b4 + ( uu-r4 )*b5 + uav*b6 + vav*b7 + wav*b8 - c2*b1/gm1;

      aj    = 0.5*rr;
      plar  = plft + prht;
      eplft = elft + plft;
      eprht = erht + prht;
      //fssub = rr*r4;
      //fssub = 0.0;
      f1 = aj*(  rlft*uulft +  rrht*uurht           - aq1 );
      f2 = aj*( rulft*uulft + rurht*uurht + r1*plar - aq2 );
      f3 = aj*( rvlft*uulft + rvrht*uurht + r2*plar - aq3 );
      f4 = aj*( rwlft*uulft + rwrht*uurht + r3*plar - aq4 );
      f5 = aj*( eplft*uulft + eprht*uurht - r4*plar - aq5 );
    }
}
//!------------------------------------------------------------------------------
//!#     Computes flux Jacobian matrix for subroutine "flux_roe" that can be
//!#               used in Newton solvers and adjoint codes.
//!#
//!#     nxyz[3]             - Three components of face normal vector.
//!#                           These can be dimensional (i.e including face area)
//!#                           or non-dimensional.
//!#                           Returned flux does not include face area.
//!#     ql[5],qr[5]         - Conserved variables (ro, ro*u, ro*v, ro*w, Et)
//!#     lmat[5][5],rmat[5][5] - Left and right state flux Jacobian matrices
//!#     gam                 - Ratio of specific heats
//!#     imode               - 0 = approximate linearization where the eigenvalues
//!#                               are treated as constants
//!#                           1 = exact linearization
//!#
//!------------------------------------------------------------------------------
//!GPU Device code equivalant C++/CUDA
//!11/19/2012
//!Dominic Chandar
//!------------------------------------------------------------------------------

FVSAND_GPU_DEVICE
void computeJacobian( real& ql1, real& ql2, real& ql3, real& ql4, real& ql5,
                      real& qr1, real& qr2, real& qr3, real& qr4, real& qr5,
                      real& nxd, real& nyd, real& nzd,
                      int & nWbfaces, int & TID,
                      real lmat[5][5], real rmat[5][5], int & rcell)    
{

real gam=1.4;
//int imode=1;


real gm1;
real area,nx,ny,nz;
real rol,ul,vl,wl,pl,hl;
real ror,ur,vr,wr,pr,hr;
real uconl,uconr;
real ubar,vbar,wbar,hbar,uconbar,cbar,robar;
//real dp,dro,du,dv,dw;
real eig1,eig2,eig3;

real fact,A,B,term1,term2,del1,del2;

real dro_dql[5],dro_dqr[5];
real du_dql[5],du_dqr[5];
real dv_dql[5],dv_dqr[5];
real dw_dql[5],dw_dqr[5];
real dp_dql[5],dp_dqr[5];
real ducon_dql[5],ducon_dqr[5];
real ddel1_dql[5],ddel1_dqr[5];
real ddel2_dql[5],ddel2_dqr[5];

real dq5_dql[5],dq5_dqr[5];
real dh_dql[5],dh_dqr[5];
real dfact_dql[5],dfact_dqr[5];
real dA_dql[5],dA_dqr[5];
real dB_dql[5],dB_dqr[5];
real drobar_dql[5],dubar_dql[5],dvbar_dql[5],dwbar_dql[5];
real drobar_dqr[5],dubar_dqr[5],dvbar_dqr[5],dwbar_dqr[5];
real dhbar_dql[5],duconbar_dql[5],dcbar_dql[5];
real dhbar_dqr[5],duconbar_dqr[5],dcbar_dqr[5];

real deig1_dql[5],deig2_dql[5],deig3_dql[5];
real deig1_dqr[5],deig2_dqr[5],deig3_dqr[5];
real dterm1_dql[5],dterm1_dqr[5];
real dterm2_dql[5],dterm2_dqr[5];
real imat[5][5];
//real cl,cr,dc_dql[5],dc_dqr[5];
//real dro,du,dv,dw,dp;
real dp;
//real t1a,t1b,t2a,t2b,t3a,t3b;

real lmat1[5][5],rmat1[5][5];
//------------------------------------------------------------------------------
      gm1 = gam - 1.0;
      area = sqrt(nxd*nxd + nyd*nyd + nzd*nzd);

      nx = nxd/area;
      ny = nyd/area;
      nz = nzd/area;

      //------> back calculate primitive state

      rol = ql1;
      ul  = ql2/ql1;
      vl  = ql3/ql1;
      wl  = ql4/ql1;
      pl  = gm1*( ql5 - 0.5 * rol * (ul*ul + vl*vl + wl*wl) );
      hl  = (ql5 + pl)/rol;
      //cl  = sqrt(gam*pl/rol);
        
      ror = qr1;
      ur  = qr2/qr1;
      vr  = qr3/qr1;
      wr  = qr4/qr1;
      pr  = gm1*( qr5 - 0.5 * ror * (ur*ur + vr*vr + wr*wr) );
      hr  = (qr5 + pr)/ror;
      //cr  = sqrt(gam*pr/ror);

      //-----> primitive state differences

      //dro = ror - rol;
      //du  =  ur - ul;
      //dv  =  vr - vl;
      //dw  =  wr - wl;
      dp  =  pr - pl;

      //----> face normal velocities

      uconr = ur*nx + vr*ny + wr*nz;
      uconl = ul*nx + vl*ny + wl*nz;
//------------------------------------------------------------------------------!
//-------> linearization of left and right primitive states <-------------------!
//------------------------------------------------------------------------------!
      //---> left state

      dro_dql[0]    = 0.0;
      dro_dql[1]    = 0.0;
      dro_dql[2]    = 0.0;
      dro_dql[3]    = 0.0;
      dro_dql[4]    = 0.0;


      dro_dql[0] = 1.0;

      du_dql[0] =  0.0;
      du_dql[1] =  0.0;
      du_dql[2] =  0.0;
      du_dql[3] =  0.0;
      du_dql[4] =  0.0;

      du_dql[0] = -ul /rol;
      du_dql[1] =  1.0/rol;

      dv_dql[0] =  0.0;
      dv_dql[1] =  0.0;
      dv_dql[2] =  0.0;
      dv_dql[3] =  0.0;
      dv_dql[4] =  0.0;
      dv_dql[0] = -vl /rol;
      dv_dql[2] =  1.0/rol;

      dw_dql[0] =  0.0;
      dw_dql[1] =  0.0;
      dw_dql[2] =  0.0;
      dw_dql[3] =  0.0;
      dw_dql[4] =  0.0;


      dw_dql[0] = -wl /rol;
      dw_dql[3] =  1.0/rol;

      dp_dql[0] =  0.5*gm1*( ul*ul + vl*vl + wl*wl );
      dp_dql[1] = -gm1*ul;
      dp_dql[2] = -gm1*vl;
      dp_dql[3] = -gm1*wl;
      dp_dql[4] =  gm1;

      dq5_dql[0] = 0.0;
      dq5_dql[1] = 0.0;
      dq5_dql[2] = 0.0;
      dq5_dql[3] = 0.0;
      dq5_dql[4] = 0.0;
      dq5_dql[4] = 1.0;

      dh_dql[0] = -(ql5 + pl)*dro_dql[0]/rol/rol + (1.0/rol)*(dq5_dql[0] + dp_dql[0]);
      dh_dql[1] = -(ql5 + pl)*dro_dql[1]/rol/rol + (1.0/rol)*(dq5_dql[1] + dp_dql[1]);
      dh_dql[2] = -(ql5 + pl)*dro_dql[2]/rol/rol + (1.0/rol)*(dq5_dql[2] + dp_dql[2]);
      dh_dql[3] = -(ql5 + pl)*dro_dql[3]/rol/rol + (1.0/rol)*(dq5_dql[3] + dp_dql[3]);
      dh_dql[4] = -(ql5 + pl)*dro_dql[4]/rol/rol + (1.0/rol)*(dq5_dql[4] + dp_dql[4]);
 
      // dc_dql[0] = (0.5*gam/cl)*( (1.0/rol)*dp_dql[0] - (pl/rol/rol)*dro_dql[0] );
      // dc_dql[1] = (0.5*gam/cl)*( (1.0/rol)*dp_dql[1] - (pl/rol/rol)*dro_dql[1] );
      // dc_dql[2] = (0.5*gam/cl)*( (1.0/rol)*dp_dql[2] - (pl/rol/rol)*dro_dql[2] );
      // dc_dql[3] = (0.5*gam/cl)*( (1.0/rol)*dp_dql[3] - (pl/rol/rol)*dro_dql[3] );
      // dc_dql[4] = (0.5*gam/cl)*( (1.0/rol)*dp_dql[4] - (pl/rol/rol)*dro_dql[4] );

      ducon_dql[0] = -uconl/rol;
      ducon_dql[1] =  nx   /rol;
      ducon_dql[2] =  ny   /rol;
      ducon_dql[3] =  nz   /rol;
      ducon_dql[4] =  0.0;

//------------------------------------------------------------------------------!
      //---> right state

      dro_dqr[0]    = 0.0;
      dro_dqr[1]    = 0.0;
      dro_dqr[2]    = 0.0;
      dro_dqr[3]    = 0.0;
      dro_dqr[4]    = 0.0;
      dro_dqr[0]    = 1.0;

      du_dqr[0] =  0.0;
      du_dqr[1] =  0.0;
      du_dqr[2] =  0.0;
      du_dqr[3] =  0.0;
      du_dqr[4] =  0.0;


      du_dqr[0] = -ur /ror;
      du_dqr[1] =  1.0/ror;

      dv_dqr[0] =  0.0;
      dv_dqr[1] =  0.0;
      dv_dqr[2] =  0.0;
      dv_dqr[3] =  0.0;
      dv_dqr[4] =  0.0;

      dv_dqr[0] = -vr /ror;
      dv_dqr[2] =  1.0/ror;

      dw_dqr[0] =  0.0;
      dw_dqr[1] =  0.0;
      dw_dqr[2] =  0.0;
      dw_dqr[3] =  0.0;
      dw_dqr[4] =  0.0;

      dw_dqr[0] = -wr /ror;
      dw_dqr[3] =  1.0/ror;

      dp_dqr[0] =  0.5*gm1*( ur*ur + vr*vr + wr*wr);
      dp_dqr[1] = -gm1*ur;
      dp_dqr[2] = -gm1*vr;
      dp_dqr[3] = -gm1*wr;
      dp_dqr[4] =  gm1;

      dq5_dqr[0] = 0.0;
      dq5_dqr[1] = 0.0;
      dq5_dqr[2] = 0.0;
      dq5_dqr[3] = 0.0;
      dq5_dqr[4] = 0.0;
      dq5_dqr[4] = 1.0;

      dh_dqr[0] = -(qr5 + pr)*dro_dqr[0]/ror/ror + (1.0/ror)*(dq5_dqr[0] + dp_dqr[0]);
      dh_dqr[1] = -(qr5 + pr)*dro_dqr[1]/ror/ror + (1.0/ror)*(dq5_dqr[1] + dp_dqr[1]);
      dh_dqr[2] = -(qr5 + pr)*dro_dqr[2]/ror/ror + (1.0/ror)*(dq5_dqr[2] + dp_dqr[2]);
      dh_dqr[3] = -(qr5 + pr)*dro_dqr[3]/ror/ror + (1.0/ror)*(dq5_dqr[3] + dp_dqr[3]);
      dh_dqr[4] = -(qr5 + pr)*dro_dqr[4]/ror/ror + (1.0/ror)*(dq5_dqr[4] + dp_dqr[4]);

      //      dc_dqr[0] = (0.5*gam/cr)*( (1.0/ror)*dp_dqr[0] - (pr/ror/ror)*dro_dqr[0] );
      // dc_dqr[1] = (0.5*gam/cr)*( (1.0/ror)*dp_dqr[1] - (pr/ror/ror)*dro_dqr[1] );
      // dc_dqr[2] = (0.5*gam/cr)*( (1.0/ror)*dp_dqr[2] - (pr/ror/ror)*dro_dqr[2] );
      // dc_dqr[3] = (0.5*gam/cr)*( (1.0/ror)*dp_dqr[3] - (pr/ror/ror)*dro_dqr[3] );
      // dc_dqr[4] = (0.5*gam/cr)*( (1.0/ror)*dp_dqr[4] - (pr/ror/ror)*dro_dqr[4] );

      ducon_dqr[0] = -uconr/ror;
      ducon_dqr[1] =  nx   /ror;
      ducon_dqr[2] =  ny   /ror;
      ducon_dqr[3] =  nz   /ror;
      ducon_dqr[4] =  0.0;

//!------------------------------------------------------------------------------!
//!----------------------------> Roe average state <-----------------------------!
//------------------------------------------------------------------------------!

      fact = sqrt(ror/rol);

      A    = 1.0 /(1.0 + fact);
      B    = fact/(1.0 + fact);

      robar = rol*fact;
      ubar  = ul*A + ur*B;
      vbar  = vl*A + vr*B;
      wbar  = wl*A + wr*B;
      hbar  = hl*A + hr*B;
      cbar = gm1*(hbar - 0.5*(ubar*ubar + vbar*vbar + wbar*wbar));
      cbar = sqrt(cbar);
      uconbar = ubar*nx + vbar*ny + wbar*nz;

//!------------------------------------------------------------------------------!
//!--------------------------> Eigenvalues <-------------------------------------!
//!------------------------------------------------------------------------------!

      eig1 = abs(uconbar);
      eig2 = abs(uconbar + cbar);
      eig3 = abs(uconbar - cbar);

//!------------------------------------------------------------------------------!
//!--------------> approximate linearization section <---------------------------!
//!------------------------------------------------------------------------------!

/*
if( imode==1 ) then
      term1 = -eig1 + 0.5*(eig2 + eig3)
      term2 = 0.5*(eig2 - eig3)
      del1  = term1*dp/cbar/cbar + term2*robar*(uconr - uconl)/cbar
      del2  = term1*(uconr - uconl)*robar + term2*dp/cbar

      ddel1_dql(:) = - term1*dp_dql(:)/cbar/cbar - term2*robar*ducon_dql(:)/cbar
      ddel1_dqr(:) = + term1*dp_dqr(:)/cbar/cbar + term2*robar*ducon_dqr(:)/cbar

      ddel2_dql(:) = - term1*ducon_dql(:)*robar - term2*dp_dql(:)/cbar
      ddel2_dqr(:) = + term1*ducon_dqr(:)*robar + term2*dp_dqr(:)/cbar

      goto 111
     
endif        
*/

//!------------------------------------------------------------------------------!
//!-----------> linearization of Roe averaged state <----------------------------!
//!------------------------------------------------------------------------------!
#pragma unroll 5
for( int i = 0; i < 5 ; i++ )
 {
      dfact_dql[i] = (0.5/fact)*(-ror/rol/rol)*dro_dql[i];
      dfact_dqr[i] = (0.5/fact)*(1.0/rol)*dro_dqr[i];

      dA_dql[i] = -dfact_dql[i]/(1.0+fact)/(1.0+fact);
      dA_dqr[i] = -dfact_dqr[i]/(1.0+fact)/(1.0+fact);

      dB_dql[i] = dfact_dql[i]/(1.0 + fact)/(1.0 + fact);
      dB_dqr[i] = dfact_dqr[i]/(1.0 + fact)/(1.0 + fact);

      drobar_dql[i] = dro_dql[i]*fact + rol*dfact_dql[i];
      drobar_dqr[i] =                   rol*dfact_dqr[i];

      dubar_dql[i] = du_dql[i]*A + ul*dA_dql[i]               + ur*dB_dql[i];
      dubar_dqr[i] =               ul*dA_dqr[i] + du_dqr[i]*B + ur*dB_dqr[i];

      dvbar_dql[i] = dv_dql[i]*A + vl*dA_dql[i]               + vr*dB_dql[i];
      dvbar_dqr[i] =               vl*dA_dqr[i] + dv_dqr[i]*B + vr*dB_dqr[i];

      dwbar_dql[i] = dw_dql[i]*A + wl*dA_dql[i]               + wr*dB_dql[i];
      dwbar_dqr[i] =               wl*dA_dqr[i] + dw_dqr[i]*B + wr*dB_dqr[i];

      dhbar_dql[i] = dh_dql[i]*A + hl*dA_dql[i]               + hr*dB_dql[i];
      dhbar_dqr[i] =               hl*dA_dqr[i] + dh_dqr[i]*B + hr*dB_dqr[i];

      dcbar_dql[i] = gm1*( dhbar_dql[i] - ubar*dubar_dql[i]     
                                        - vbar*dvbar_dql[i]     
                                        - wbar*dwbar_dql[i] );
      dcbar_dql[i] = dcbar_dql[i]*0.5/cbar;

      dcbar_dqr[i] = gm1*( dhbar_dqr[i] - ubar*dubar_dqr[i]     
                                        - vbar*dvbar_dqr[i]     
                                        - wbar*dwbar_dqr[i] );
      dcbar_dqr[i] = dcbar_dqr[i]*0.5/cbar ;

      duconbar_dql[i] = dubar_dql[i]*nx + dvbar_dql[i]*ny + dwbar_dql[i]*nz;
      duconbar_dqr[i] = dubar_dqr[i]*nx + dvbar_dqr[i]*ny + dwbar_dqr[i]*nz;

//!------------------------------------------------------------------------------!
//!------------------> linearization of Eigenvalues <----------------------------!
//!------------------------------------------------------------------------------!

      if(uconbar>=0.0) 
        {
                deig1_dql[i] = duconbar_dql[i];
                deig1_dqr[i] = duconbar_dqr[i];
        }
      else 
        {
                deig1_dql[i] = -duconbar_dql[i];
                deig1_dqr[i] = -duconbar_dqr[i];
        }

      if( (uconbar + cbar) >= 0.0 ) 
        {
                deig2_dql[i] = ( duconbar_dql[i] + dcbar_dql[i] );
                deig2_dqr[i] = ( duconbar_dqr[i] + dcbar_dqr[i] );
        }
      else
        {
                deig2_dql[i] = -( duconbar_dql[i] + dcbar_dql[i] );
                deig2_dqr[i] = -( duconbar_dqr[i] + dcbar_dqr[i] );
        }

      if( (uconbar - cbar) >= 0.0 )
        {
                deig3_dql[i] = ( duconbar_dql[i] - dcbar_dql[i] );
                deig3_dqr[i] = ( duconbar_dqr[i] - dcbar_dqr[i] );
        }
      else
        {
                deig3_dql[i] = -( duconbar_dql[i] - dcbar_dql[i] );
                deig3_dqr[i] = -( duconbar_dqr[i] - dcbar_dqr[i] );
        }

//!------------------------------------------------------------------------------!
      term1 = -eig1 + 0.5*(eig2 + eig3);
      term2 = 0.5*(eig2 - eig3);
      del1  = term1*dp/cbar/cbar + term2*robar*(uconr - uconl)/cbar;
      del2  = term1*(uconr - uconl)*robar + term2*dp/cbar;

      dterm1_dql[i] = -deig1_dql[i] + 0.5*( deig2_dql[i] + deig3_dql[i] );
      dterm1_dqr[i] = -deig1_dqr[i] + 0.5*( deig2_dqr[i] + deig3_dqr[i] );

      dterm2_dql[i] = 0.5*( deig2_dql[i] - deig3_dql[i] );
      dterm2_dqr[i] = 0.5*( deig2_dqr[i] - deig3_dqr[i] );

      ddel1_dql[i] = dterm1_dql[i]*dp/cbar/cbar - term1*dp_dql[i]/cbar/cbar - 2.0*term1*dp*dcbar_dql[i]/cbar/cbar/cbar;
      ddel1_dql[i] = ddel1_dql[i] + dterm2_dql[i]*robar*( uconr-uconl )/cbar + term2*drobar_dql[i]*(uconr-uconl)/cbar 
                                  - term2*robar*ducon_dql[i]/cbar - dcbar_dql[i]*term2*robar*(uconr-uconl)/cbar/cbar;

      ddel1_dqr[i] = dterm1_dqr[i]*dp/cbar/cbar + term1*dp_dqr[i]/cbar/cbar - 2.0*term1*dp*dcbar_dqr[i]/cbar/cbar/cbar;
      ddel1_dqr[i] = ddel1_dqr[i] + dterm2_dqr[i]*robar*( uconr-uconl )/cbar + term2*drobar_dqr[i]*(uconr-uconl)/cbar 
                                  + term2*robar*ducon_dqr[i]/cbar - dcbar_dqr[i]*term2*robar*(uconr-uconl)/cbar/cbar;

      ddel2_dql[i] = dterm1_dql[i]*(uconr-uconl)*robar - term1*ducon_dql[i]*robar + term1*(uconr-uconl)*drobar_dql[i];
      ddel2_dql[i] = ddel2_dql[i] + dterm2_dql[i]*dp/cbar - term2*dp_dql[i]/cbar - dcbar_dql[i]*term2*dp/cbar/cbar;

      ddel2_dqr[i] = dterm1_dqr[i]*(uconr-uconl)*robar + term1*ducon_dqr[i]*robar + term1*(uconr-uconl)*drobar_dqr[i];
      ddel2_dqr[i] = ddel2_dqr[i] + dterm2_dqr[i]*dp/cbar + term2*dp_dqr[i]/cbar - dcbar_dqr[i]*term2*dp/cbar/cbar;
}

//!------------------------------------------------------------------------------!
//111 continue
//!------------------------------------------------------------------------------!
//!-----------------------> Roe flux Jacobian <----------------------------------!
//!------------------------------------------------------------------------------!


//      !------------> common linearization terms

      FOR2(i,5,j,5)
       {
        lmat[i][j] = 0.0;
        rmat[i][j] = 0.0;
        imat[i][j] = 0.0;
       }

      for ( int i = 0 ; i < 5 ; i++ )
           imat[i][i] = 1.0;
     
      FOR2(i,5,j,5)
       {      
         lmat[i][j] = lmat[i][j] - eig1*imat[i][j];
         rmat[i][j] = rmat[i][j] + eig1*imat[i][j];
       }

      #pragma unroll 5
      for ( int i = 0 ; i < 5 ; i++ )
      {
       lmat[0][i] = lmat[0][i] + ddel1_dql[i];
       rmat[0][i] = rmat[0][i] + ddel1_dqr[i];

       lmat[1][i] = lmat[1][i] + ddel1_dql[i]*ubar + ddel2_dql[i]*nx;
       rmat[1][i] = rmat[1][i] + ddel1_dqr[i]*ubar + ddel2_dqr[i]*nx;

       lmat[2][i] = lmat[2][i] + ddel1_dql[i]*vbar + ddel2_dql[i]*ny;
       rmat[2][i] = rmat[2][i] + ddel1_dqr[i]*vbar + ddel2_dqr[i]*ny;

       lmat[3][i] = lmat[3][i] + ddel1_dql[i]*wbar + ddel2_dql[i]*nz;
       rmat[3][i] = rmat[3][i] + ddel1_dqr[i]*wbar + ddel2_dqr[i]*nz;

       lmat[4][i] = lmat[4][i] + ddel1_dql[i]*hbar + ddel2_dql[i]*uconbar;
       rmat[4][i] = rmat[4][i] + ddel1_dqr[i]*hbar + ddel2_dqr[i]*uconbar;
       }
    

      //------> additional terms for exact linearization

     // if(imode/=1) then
        #pragma unroll 5
         for ( int j = 0 ; j < 5 ; j++ )
         {       
           lmat[0][j] = lmat[0][j] + ( qr1 - ql1 )* deig1_dql[j];  rmat[0][j] = rmat[0][j] + ( qr1 - ql1 )* deig1_dqr[j];
           lmat[1][j] = lmat[1][j] + ( qr2 - ql2 )* deig1_dql[j];  rmat[1][j] = rmat[1][j] + ( qr2 - ql2 )* deig1_dqr[j];
           lmat[2][j] = lmat[2][j] + ( qr3 - ql3 )* deig1_dql[j];  rmat[2][j] = rmat[2][j] + ( qr3 - ql3 )* deig1_dqr[j];
           lmat[3][j] = lmat[3][j] + ( qr4 - ql4 )* deig1_dql[j];  rmat[3][j] = rmat[3][j] + ( qr4 - ql4 )* deig1_dqr[j];
           lmat[4][j] = lmat[4][j] + ( qr5 - ql5 )* deig1_dql[j];  rmat[4][j] = rmat[4][j] + ( qr5 - ql5 )* deig1_dqr[j];

         }         
         
         #pragma unroll 5       
         for ( int j = 0 ; j < 5 ; j++ )
          {
            lmat[1][j] = lmat[1][j] + del1*dubar_dql[j];
            rmat[1][j] = rmat[1][j] + del1*dubar_dqr[j];

            lmat[2][j] = lmat[2][j] + del1*dvbar_dql[j];
            rmat[2][j] = rmat[2][j] + del1*dvbar_dqr[j];

            lmat[3][j] = lmat[3][j] + del1*dwbar_dql[j];
            rmat[3][j] = rmat[3][j] + del1*dwbar_dqr[j];

            lmat[4][j] = lmat[4][j] + del1*dhbar_dql[j] + del2*duconbar_dql[j];
            rmat[4][j] = rmat[4][j] + del1*dhbar_dqr[j] + del2*duconbar_dqr[j];
          }
    //  endif

//      !------------------------------------------------------------------------------!
//      !-------------------------------------------------------!
//      !-----------> Compute native flux Jacobian <------------!
//      !-------------------------------------------------------!


    for ( int j = 0 ; j < 5 ; j++ )
    {
//      !-----> left state
      lmat1[0][j] = dro_dql[j]*uconl + rol*ducon_dql[j];

      lmat1[1][j] = dro_dql[j]*uconl*ul + rol*ducon_dql[j]*ul + rol*uconl*du_dql[j];
      lmat1[2][j] = dro_dql[j]*uconl*vl + rol*ducon_dql[j]*vl + rol*uconl*dv_dql[j];
      lmat1[3][j] = dro_dql[j]*uconl*wl + rol*ducon_dql[j]*wl + rol*uconl*dw_dql[j];

      lmat1[2][j] = lmat1[2][j] + ny*dp_dql[j];
      lmat1[1][j] = lmat1[1][j] + nx*dp_dql[j];
      lmat1[3][j] = lmat1[3][j] + nz*dp_dql[j];

      lmat1[4][j] = ( dq5_dql[j] + dp_dql[j] )*uconl + (ql5 + pl)*ducon_dql[j];

//      !======================================================================
//      !-----> right state

      rmat1[0][j] = dro_dqr[j]*uconr + ror*ducon_dqr[j];

      rmat1[1][j] = dro_dqr[j]*uconr*ur + ror*ducon_dqr[j]*ur + ror*uconr*du_dqr[j];
      rmat1[2][j] = dro_dqr[j]*uconr*vr + ror*ducon_dqr[j]*vr + ror*uconr*dv_dqr[j];
      rmat1[3][j] = dro_dqr[j]*uconr*wr + ror*ducon_dqr[j]*wr + ror*uconr*dw_dqr[j];

      rmat1[2][j] = rmat1[2][j] + ny*dp_dqr[j];
      rmat1[1][j] = rmat1[1][j] + nx*dp_dqr[j];
      rmat1[3][j] = rmat1[3][j] + nz*dp_dqr[j];

      rmat1[4][j] = ( dq5_dqr[j] + dp_dqr[j] )*uconr + (qr5 + pr)*ducon_dqr[j];

//      !======================================================================

//      lmat(:,:) = 0.5*( lmat1(:,:) - lmat(:,:) )
//      rmat(:,:) = 0.5*( rmat1(:,:) - rmat(:,:) )
    }
 
 if ( TID < nWbfaces )
  {
     FOR2(i,5,j,5)
      {
        lmat[i][j]=0.0;
        rmat[i][j]=0.0;
      }

     lmat[1][4] = gm1*nx;
     lmat[2][4] = gm1*ny;
     lmat[3][4] = gm1*nz;
  }
 else  
  {
     if ( rcell >= 0 )
      {
        FOR2(i,5,j,5)
        {
          lmat[i][j] = 0.5*( lmat1[i][j] - lmat[i][j] );
          rmat[i][j] = 0.5*( rmat1[i][j] - rmat[i][j] );
        }
      }
     else
      {
        FOR2(i,5,j,5)
        {
         lmat[i][j]=0.0;
         rmat[i][j]=0.0;
        }
      }
 }   



}


