FVSAND_GPU_DEVICE
void InterfaceFlux_Inviscid_fp( real & f1, real & f2, real & f3, real & f4, real & f5,
				const real & rlft, const real& ulft, const real & vlft, const real & wlft, real & plft,
				const real & rrht, const real& urht, const real & vrht, const real & wrht, real & prht,
				real & dsx, real & dsy, real & dsz,
				real & gx, real & gy, real & gz, real &spec, int & faceID)
{
  real  gm1=0.4;
  real ql2,ql3,ql4;
  real eps;
  real rlfti,rulft,rvlft,rwlft,uvwl,elft,hlft,clft;
  real qr2,qr3,qr4;
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

      ql2 =rlft*ulft;
      ql3 =rlft*vlft;
      ql4 =rlft*wlft;

      rlfti = 1.0/rlft;
      rulft = ql2;
      rvlft = ql3;
      rwlft = ql4;
      uvwl = 0.5*( ulft*ulft + vlft*vlft + wlft*wlft );
      elft = plft/gm1 + rlft*uvwl;
      hlft = ( elft + plft )*rlfti;
      clft = sqrt( gm1*( hlft - uvwl ) );

      qr2=rrht*urht;
      qr3=rrht*vrht;
      qr4=rrht*wrht;

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

FVSAND_GPU_DEVICE
void computeResidualJacobianDiag_fp( real_d & f1, real_d & f2, real_d & f3, real_d & f4, real_d & f5,
				     const real_d & rlft, const real_d& ulft, const real_d & vlft, const real_d & wlft, const real_d & plft,
				     const real_d & rrht, const real_d& urht, const real_d & vrht, const real_d & wrht, const real_d & prht,
				     const real_d & dsx, const real_d & dsy, const real_d & dsz,
				     const real_d & gx, const real_d & gy, const real_d & gz, const int & faceID,
				     real_j *lmatout, real_j scaling, const int idx, const int ncells)
{
  real_d  gm1=0.4;
  real_d ql1,ql2,ql3,ql4,ql5;
  real_d eps;
  real_d rlfti,rulft,rvlft,rwlft,uvwl,elft,hlft,clft;
  real_d qr1,qr2,qr3,qr4,qr5;
  real_d rrhti,rurht,rvrht,rwrht,uvwr,erht,hrht,crht;
  real_d rat,rati,rav,uav,vav,wav,hav,uvw,cav;
  real_d aq1,aq2,aq3,aq4,aq5,ri1,ri2,ri3,ri4,rr2,rr,r0,r1,r2,r3,r4;
  real_d uu,c2,c2i,auu,aupc,aumc,uulft,uurht,upclft,upcrht;
  real_d umclft,umcrht,dauu,dauus,daupc,daumc,daumcs,rcav,aquu;
  real_d daupcs,c2ih,ruuav,b1,b2,b3,b4,b5,b6,b7,b8,aj;
  real_d plar,eplft,eprht;
  real_d faceSpeed;


  real_j term1,term2,del1,del2;
  real_j dro_dql[5];
  real_j du_dql[5];
  real_j dv_dql[5];
  real_j dw_dql[5];
  real_j dp_dql[5];
  real_j ducon_dql[5];
  real_j ddel1_dql[5];
  real_j ddel2_dql[5];
  
  real_j dq5_dql[5];
  real_j dh_dql[5];
  real_j dfact_dql[5];
  real_j dA_dql[5];
  real_j dB_dql[5];
  real_j drobar_dql[5],dubar_dql[5],dvbar_dql[5],dwbar_dql[5];
  real_j dhbar_dql[5],duconbar_dql[5],dcbar_dql[5];
  
  real_j deig1_dql[5],deig2_dql[5],deig3_dql[5];
  real_j dterm1_dql[5];
  real_j dterm2_dql[5];
  real_j imat[5][5];
  
  real_j lmat[5][5];
  real_j lmat1[5][5];

  //
  // first executable statement
  //
  eps = 1.e-6;
 
  faceSpeed = (gx*dsx + gy*dsy + gz*dsz)/sqrt(dsx*dsx + dsy*dsy + dsz*dsz);
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

  if ( faceID == -2 ) {// boundary face
    //plft = gm1*(ql5 - 0.5*(ql2*ql2 + ql3*ql3 + ql4*ql4)/ql1);
      f1 = 0.0;
      f2 = plft*dsx;
      f3 = plft*dsy;
      f4 = plft*dsz;
      f5 = 0.0;

      //ulft = ql2/ql1;
      //vlft = ql3/ql1;
      //wlft = ql4/ql1;
      uvwl = ulft*ulft + vlft*vlft + wlft*wlft;

     for(int n = 0; n<5; n++){
        lmat[0][n] = 0.0;
        lmat[4][n] = 0.0;
     }
     lmat[1][0] = gm1*0.5-uvwl*r1;
     lmat[1][1] = -gm1*ulft*r1;
     lmat[1][2] = -gm1*vlft*r1;
     lmat[1][3] = -gm1*wlft*r1;
     lmat[1][4] = gm1*r1;

     lmat[2][0] = gm1*0.5*uvwl*r2;
     lmat[2][1] = -gm1*ulft*r2;
     lmat[2][2] = -gm1*vlft*r2;
     lmat[2][3] = -gm1*wlft*r2;
     lmat[2][4] = gm1*r2;

     lmat[3][0] = gm1*0.5*uvwl*r3;
     lmat[3][1] = -gm1*ulft*r3;
     lmat[3][2] = -gm1*vlft*r3;
     lmat[3][3] = -gm1*wlft*r3;
     lmat[3][4] = gm1*r3;
     FOR2(i,5,j,5)
     {
       lmat[i][j] = lmat[i][j]*rr;
     }
  }
  else {
//------------------------------------------------------------------------------!
      //---> left state
      ql1=rlft;
      ql2=rlft*ulft;
      ql3=rlft*vlft;
      ql4=rlft*wlft;
      ql5=(plft/gm1 + 0.5*(ql2*ql2 + ql3*ql3 + ql4*ql4)/ql1);

      rlfti = 1.0/rlft;
      rulft = ql2;
      rvlft = ql3;
      rwlft = ql4;
      uvwl = 0.5*( ulft*ulft + vlft*vlft + wlft*wlft );
      elft = plft/gm1 + rlft*uvwl;
      hlft = ( elft + plft )*rlfti;
      clft = sqrt( gm1*( hlft - uvwl ) );

      uulft = r1*ulft + r2*vlft + r3*wlft + r4;

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

      du_dql[0] = -ulft /rlft;
      du_dql[1] =  1.0/rlft;

      dv_dql[0] =  0.0;
      dv_dql[1] =  0.0;
      dv_dql[2] =  0.0;
      dv_dql[3] =  0.0;
      dv_dql[4] =  0.0;

      dv_dql[0] = -vlft /rlft;
      dv_dql[2] =  1.0/rlft;

      dw_dql[0] =  0.0;
      dw_dql[1] =  0.0;
      dw_dql[2] =  0.0;
      dw_dql[3] =  0.0;
      dw_dql[4] =  0.0;

      dw_dql[0] = -wlft /rlft;
      dw_dql[3] =  1.0/rlft;

      dp_dql[0] =  0.5*gm1*( ulft*ulft + vlft*vlft + wlft*wlft );
      dp_dql[1] = -gm1*ulft;
      dp_dql[2] = -gm1*vlft;
      dp_dql[3] = -gm1*wlft;
      dp_dql[4] =  gm1;

      dq5_dql[0] = 0.0;
      dq5_dql[1] = 0.0;
      dq5_dql[2] = 0.0;
      dq5_dql[3] = 0.0;
      dq5_dql[4] = 0.0;

      dq5_dql[4] = 1.0;

      dh_dql[0] = -(ql5 + plft)*dro_dql[0]/rlft/rlft + (1.0/rlft)*(dq5_dql[0] + dp_dql[0]);
      dh_dql[1] = -(ql5 + plft)*dro_dql[1]/rlft/rlft + (1.0/rlft)*(dq5_dql[1] + dp_dql[1]);
      dh_dql[2] = -(ql5 + plft)*dro_dql[2]/rlft/rlft + (1.0/rlft)*(dq5_dql[2] + dp_dql[2]);
      dh_dql[3] = -(ql5 + plft)*dro_dql[3]/rlft/rlft + (1.0/rlft)*(dq5_dql[3] + dp_dql[3]);
      dh_dql[4] = -(ql5 + plft)*dro_dql[4]/rlft/rlft + (1.0/rlft)*(dq5_dql[4] + dp_dql[4]);
 
      ducon_dql[0] = -uulft/rlft;
      ducon_dql[1] =  r1   /rlft;
      ducon_dql[2] =  r2   /rlft;
      ducon_dql[3] =  r3   /rlft;
      ducon_dql[4] =  0.0;

//------------------------------------------------------------------------------!
      //---> right state

      qr1=rrht;
      qr2=rrht*urht;
      qr3=rrht*vrht;
      qr4=rrht*wrht;
      qr5=(prht/gm1+0.5*(qr2*qr2 + qr3*qr3 + qr4*qr4)/qr1);
	
      rrhti = 1.0/rrht;
      rurht = qr2;
      rvrht = qr3;
      rwrht = qr4;
      uvwr = 0.5*( urht*urht + vrht*vrht + wrht*wrht );
      erht = prht/gm1 + rrht*uvwr;
      hrht = ( erht + prht )*rrhti;
      crht = sqrt( gm1*( hrht - uvwr ) );

      uurht = r1*urht + r2*vrht + r3*wrht + r4;

      aq1  = rrht - rlft;
      aq2  = urht - ulft;
      aq3  = vrht - vlft;
      aq4  = wrht - wlft;
      aq5  = prht - plft;

//------------------------------------------------------------------------------!
      //---> Roe average state

      rat  = sqrt( rrht*rlfti );
      rati = 1.0/( rat + 1. );
      rav  =   rat*rlft;
      uav  = ( rat*urht + ulft )*rati;
      vav  = ( rat*vrht + vlft )*rati;
      wav  = ( rat*wrht + wlft )*rati;
      hav  = ( rat*hrht + hlft )*rati;
      uvw  = 0.5*( uav*uav + vav*vav + wav*wav );
      cav  = sqrt( gm1*( hav - uvw ) );
      uu  = r1*uav + r2*vav + r3*wav + r4;

      auu   = abs( uu    );
      aupc  = abs( uu+cav );
      aumc  = abs( uu-cav );

#pragma unroll 5
for( int i = 0; i < 5 ; i++ )
 {
      dfact_dql[i] = (0.5/rat)*(-rrht/rlft/rlft)*dro_dql[i];
      dA_dql[i] = -dfact_dql[i]/(1.0+rat)/(1.0+rat);
      dB_dql[i] = dfact_dql[i]/(1.0 + rat)/(1.0 + rat);

      drobar_dql[i] = dro_dql[i]*rat + rlft*dfact_dql[i];
      dubar_dql[i] = du_dql[i]*rati + ulft*dA_dql[i]               + urht*dB_dql[i];
      dvbar_dql[i] = dv_dql[i]*rati + vlft*dA_dql[i]               + vrht*dB_dql[i];
      dwbar_dql[i] = dw_dql[i]*rati + wlft*dA_dql[i]               + wrht*dB_dql[i];
      dhbar_dql[i] = dh_dql[i]*rati + hlft*dA_dql[i]               + hrht*dB_dql[i];

      dcbar_dql[i] = gm1*( dhbar_dql[i] - uav*dubar_dql[i]     
                                        - vav*dvbar_dql[i]     
                                        - wav*dwbar_dql[i] );
      dcbar_dql[i] = dcbar_dql[i]*0.5/cav;


      duconbar_dql[i] = dubar_dql[i]*r1 + dvbar_dql[i]*r2 + dwbar_dql[i]*r3;

      if(uu>=0.0) 
        {
                deig1_dql[i] = duconbar_dql[i];
        }
      else 
        {
                deig1_dql[i] = -duconbar_dql[i];
        }

      if( (uu + cav) >= 0.0 ) 
        {
                deig2_dql[i] = ( duconbar_dql[i] + dcbar_dql[i] );
        }
      else
        {
                deig2_dql[i] = -( duconbar_dql[i] + dcbar_dql[i] );
        }

      if( (uu - cav) >= 0.0 )
        {
                deig3_dql[i] = ( duconbar_dql[i] - dcbar_dql[i] );
        }
      else
        {
                deig3_dql[i] = -( duconbar_dql[i] - dcbar_dql[i] );
        }

      term1 = -auu + 0.5*(aupc + aumc);
      term2 = 0.5*(aupc - aumc);
      del1  = term1*aq5/cav/cav + term2*rav*(uurht - uulft)/cav;
      del2  = term1*(uurht - uulft)*rav + term2*aq5/cav;

      dterm1_dql[i] = -deig1_dql[i] + 0.5*( deig2_dql[i] + deig3_dql[i] );
      dterm2_dql[i] = 0.5*( deig2_dql[i] - deig3_dql[i] );

      ddel1_dql[i] = dterm1_dql[i]*aq5/cav/cav - term1*dp_dql[i]/cav/cav - 2.0*term1*aq5*dcbar_dql[i]/cav/cav/cav;
      ddel1_dql[i] = ddel1_dql[i] + dterm2_dql[i]*rav*( uurht-uulft )/cav + term2*drobar_dql[i]*(uurht-uulft)/cav 
                                  - term2*rav*ducon_dql[i]/cav - dcbar_dql[i]*term2*rav*(uurht-uulft)/cav/cav;

      ddel2_dql[i] = dterm1_dql[i]*(uurht-uulft)*rav - term1*ducon_dql[i]*rav + term1*(uurht-uulft)*drobar_dql[i];
      ddel2_dql[i] = ddel2_dql[i] + dterm2_dql[i]*aq5/cav - term2*dp_dql[i]/cav - dcbar_dql[i]*term2*aq5/cav/cav;

}

      c2  = cav*cav;
      c2i = 1.0/c2;

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

      FOR2(i,5,j,5)
       {
        lmat[i][j] = 0.0;
        lmat1[i][j] = 0.0;
        imat[i][j] = 0.0;
       }

      for ( int i = 0 ; i < 5 ; i++ )
           imat[i][i] = 1.0;
     
      FOR2(i,5,j,5)
       {      
         lmat[i][j] = lmat[i][j] - auu*imat[i][j];
       }


      #pragma unroll 5
      for ( int i = 0 ; i < 5 ; i++ )
      {
       lmat[0][i] = lmat[0][i] + ddel1_dql[i];
       lmat[1][i] = lmat[1][i] + ddel1_dql[i]*uav + ddel2_dql[i]*r1;
       lmat[2][i] = lmat[2][i] + ddel1_dql[i]*vav + ddel2_dql[i]*r2;
       lmat[3][i] = lmat[3][i] + ddel1_dql[i]*wav + ddel2_dql[i]*r3;
       lmat[4][i] = lmat[4][i] + ddel1_dql[i]*hav + ddel2_dql[i]*uu;
       }
    
      //------> additional terms for exact linearization

     // if(imode/=1) then
        #pragma unroll 5
         for ( int j = 0 ; j < 5 ; j++ )
         {       
           lmat[0][j] = lmat[0][j] + ( qr1 - ql1 )* deig1_dql[j];  
	   lmat[1][j] = lmat[1][j] + ( qr2 - ql2 )* deig1_dql[j];  
	   lmat[2][j] = lmat[2][j] + ( qr3 - ql3 )* deig1_dql[j];  
	   lmat[3][j] = lmat[3][j] + ( qr4 - ql4 )* deig1_dql[j];  
	   lmat[4][j] = lmat[4][j] + ( qr5 - ql5 )* deig1_dql[j];  
         }         
         
         #pragma unroll 5       
         for ( int j = 0 ; j < 5 ; j++ )
          {
            lmat[1][j] = lmat[1][j] + del1*dubar_dql[j];
            lmat[2][j] = lmat[2][j] + del1*dvbar_dql[j];
            lmat[3][j] = lmat[3][j] + del1*dwbar_dql[j];
            lmat[4][j] = lmat[4][j] + del1*dhbar_dql[j] + del2*duconbar_dql[j];
          }

    //  endif

//      !------------------------------------------------------------------------------!
//      !-------------------------------------------------------!
//      !-----------> Compute native flux Jacobian <------------!
//      !-------------------------------------------------------!

    for ( int j = 0 ; j < 5 ; j++ )
    {
//      !-----> left state
      lmat1[0][j] = dro_dql[j]*uulft + rlft*ducon_dql[j];

      lmat1[1][j] = dro_dql[j]*uulft*ulft + rlft*ducon_dql[j]*ulft + rlft*uulft*du_dql[j];
      lmat1[2][j] = dro_dql[j]*uulft*vlft + rlft*ducon_dql[j]*vlft + rlft*uulft*dv_dql[j];
      lmat1[3][j] = dro_dql[j]*uulft*wlft + rlft*ducon_dql[j]*wlft + rlft*uulft*dw_dql[j];

      lmat1[2][j] = lmat1[2][j] + r2*dp_dql[j];
      lmat1[1][j] = lmat1[1][j] + r1*dp_dql[j];
      lmat1[3][j] = lmat1[3][j] + r3*dp_dql[j];

      lmat1[4][j] = ( dq5_dql[j] + dp_dql[j] )*uulft + (ql5 + plft)*ducon_dql[j];

    }
//      !======================================================================
     FOR2(i,5,j,5)
     {
       lmat[i][j] = 0.5*( lmat1[i][j] - lmat[i][j] )*rr;
     }
 }  
 FOR2(i,5,j,5)
     {
	     int index0 = i*5+j;
             int index1 = ncells*index0 + idx;
	     lmatout[index1] += (lmat[i][j]*scaling);
     }
}
