module NS_mms
  !
  real*8 :: gamma=1.4d0
  real*8 :: pr=0.72d0
  real*8 :: prtr=0.9d0
  real*8 :: rey=1000d0
  real*8 :: c2b=0.3678d0
  real*8 :: gm1,ggm1,prfac,prtrfac,reyinv,TWOTHIRD
  real*8 :: eps=1d-10
  real*8 :: turmu=0d0
  !
  private  :: gamma,pr,prtr,rey,c2b,gm1,ggm1,prfac,prtrfac,reyinv,TWOTHIRD,kcond,eps,turmu
contains
  !
  subroutine getqreal(q,x,y,z)
    !
    implicit none
    !
    real*8, intent(in) :: x,y,z
    real*8, intent(out) :: q(5)
    real*8 :: rho,u,v,w,p,e
    !
    rho=1d0+0.1d0*exp(-x*x-y*y-z*z)
    u=0.1d0*sin(x*x+y*y+z*z)
    v=0.1d0*cos(x*x+y*y+z*z)
    w=0.1d0*sin(x*y+x*z+y*z)
    p=1d0/gamma+0.1d0*sin(x)*sin(y)*sin(z)
    e=p/(gamma-1)+0.5d0*rho*(u*u+v*v+w*w)
    !
    q(1)=rho
    q(2)=rho*u
    q(3)=rho*v
    q(4)=rho*w
    q(5)=e
    write(10,*) x,y,z,q(1)
  end subroutine getqreal
  !
  subroutine gradqreal(gradu,x,y,z)
    !
    implicit none
    !
    real*8, intent(out) :: gradu(3,4)
    real*8, intent(in) :: x,y,z
    !
    real*8 :: rho,u,v,w,p,wp,T
    real*8 :: drho_dx,drho_dy,drho_dz
    real*8 :: dp_dx,dp_dy,dp_dz
    real*8 :: dt_dx,dt_dy,dt_dz
    real*8 :: du_dx,du_dy,du_dz
    real*8 :: dv_dx,dv_dy,dv_dz
    real*8 :: dw_dx,dw_dy,dw_dz
    ! 
    rho=1d0+0.1d0*exp(-x*x-y*y-z*z)
    u=0.1d0*sin(x*x+y*y+z*z)
    v=0.1d0*cos(x*x+y*y+z*z)
    w=0.1d0*sin(x*y+x*z+y*z)
    p=1d0/gamma+0.1d0*sin(x)*sin(y)*sin(z)
    !call getq(q,x,y,z)
    T=gamma*p/rho
    !
    drho_dx=-2d0*x*(rho-1d0)
    drho_dy=-2d0*y*(rho-1d0)
    drho_dz=-2d0*z*(rho-1d0)
    !
    dp_dx=0.1d0*cos(x)*sin(y)*sin(z)
    dp_dy=0.1d0*sin(x)*cos(y)*sin(z)
    dp_dz=0.1d0*sin(x)*sin(y)*cos(z)
    !
    du_dx=2d0*x*v
    du_dy=2d0*y*v
    du_dz=2d0*z*v
    dt_dx=gamma*(dp_dx/rho-p*drho_dx/(rho*rho))
    !
    dv_dx=-2d0*x*u
    dv_dy=-2d0*y*u
    dv_dz=-2d0*z*u
    dt_dy=gamma*(dp_dy/rho-p*drho_dy/(rho*rho))
    !
    wp=0.1d0*cos(x*y+x*z+y*z)
    dw_dx=(y+z)*wp
    dw_dy=(x+z)*wp
    dw_dz=(x+y)*wp
    dt_dz=gamma*(dp_dz/rho-p*drho_dz/(rho*rho))
    !
    gradu(1,:)=(/du_dx,dv_dx,dw_dx,dt_dx/)
    gradu(2,:)=(/du_dy,dv_dy,dw_dy,dt_dy/)
    gradu(3,:)=(/du_dz,dv_dz,dw_dz,dt_dz/)
    !
  end subroutine gradqreal
  !
  subroutine getq(q,x,y,z)
    !
    implicit none
    !
    complex*16, intent(in) :: x,y,z
    complex*16, intent(out) :: q(5)
    complex*16 :: rho,u,v,w,p,e
    !
    rho=1d0+0.1d0*exp(-x*x-y*y-z*z)
    u=0.1d0*sin(x*x+y*y+z*z)
    v=0.1d0*cos(x*x+y*y+z*z)
    w=0.1d0*sin(x*y+x*z+y*z)
    p=1d0/gamma+0.1d0*sin(x)*sin(y)*sin(z)
    e=p/(gamma-1)+0.5d0*rho*(u*u+v*v+w*w)
    !
    q(1)=rho
    q(2)=rho*u
    q(3)=rho*v
    q(4)=rho*w
    q(5)=e
  end subroutine getq
  !
  subroutine gradq(gradu,x,y,z)
    !
    implicit none
    !
    complex*16, intent(out) :: gradu(3,4)
    complex*16, intent(in) :: x,y,z
    !
    complex*16 :: rho,u,v,w,p,wp,T
    complex*16 :: drho_dx,drho_dy,drho_dz
    complex*16 :: dp_dx,dp_dy,dp_dz
    complex*16 :: dt_dx,dt_dy,dt_dz
    complex*16 :: du_dx,du_dy,du_dz
    complex*16 :: dv_dx,dv_dy,dv_dz
    complex*16 :: dw_dx,dw_dy,dw_dz
    ! 
    rho=1d0+0.1d0*exp(-x*x-y*y-z*z)
    u=0.1d0*sin(x*x+y*y+z*z)
    v=0.1d0*cos(x*x+y*y+z*z)
    w=0.1d0*sin(x*y+x*z+y*z)
    p=1d0/gamma+0.1d0*sin(x)*sin(y)*sin(z)
    !call getq(q,x,y,z)
    T=gamma*p/rho
    !
    drho_dx=-2d0*x*(rho-1d0)
    drho_dy=-2d0*y*(rho-1d0)
    drho_dz=-2d0*z*(rho-1d0)
    !
    dp_dx=0.1d0*cos(x)*sin(y)*sin(z)
    dp_dy=0.1d0*sin(x)*cos(y)*sin(z)
    dp_dz=0.1d0*sin(x)*sin(y)*cos(z)
    !
    du_dx=2d0*x*v
    du_dy=2d0*y*v
    du_dz=2d0*z*v
    dt_dx=gamma*(dp_dx/rho-p*drho_dx/(rho*rho))
    !
    dv_dx=-2d0*x*u
    dv_dy=-2d0*y*u
    dv_dz=-2d0*z*u
    dt_dy=gamma*(dp_dy/rho-p*drho_dy/(rho*rho))
    !
    wp=0.1d0*cos(x*y+x*z+y*z)
    dw_dx=(y+z)*wp
    dw_dy=(x+z)*wp
    dw_dz=(x+y)*wp
    dt_dz=gamma*(dp_dz/rho-p*drho_dz/(rho*rho))
    !
    gradu(1,:)=(/du_dx,dv_dx,dw_dx,dt_dx/)
    gradu(2,:)=(/du_dy,dv_dy,dw_dy,dt_dy/)
    gradu(3,:)=(/du_dz,dv_dz,dw_dz,dt_dz/)
    !
  end subroutine gradq
  !
  subroutine inviscidflux(idir,flux,xx)
    !
    implicit none
    integer, intent(in) :: idir
    complex*16, intent(out) :: flux(5)
    complex*16,intent(in) :: xx(3)
    !
    complex*16 :: q(5)
    complex*16 :: p
    complex*16 :: v
    !
    call getq(q,xx(1),xx(2),xx(3))
    p=(gamma-1)*(q(5)-0.5d0*(q(2)*q(2)+q(3)*q(3)+q(4)*q(4))/q(1))
    v=q(idir+1)/q(1)
    flux(1)=q(1)*v
    flux(2)=q(2)*v
    flux(3)=q(3)*v
    flux(4)=q(4)*v
    flux(5)=(q(5)+p)*v
    flux(idir+1)=flux(idir+1)+p
    !
  end subroutine inviscidflux
  !
  subroutine viscousFlux(idir,vflux,xx,idebug)
    !
    implicit none
    integer :: idir
    complex*16, intent(out) :: vflux(5)
    complex*16, intent(in) :: xx(3)
    integer, intent(in) :: idebug
    !
    complex*16 :: q(5)
    complex*16 :: p,T,mu
    complex*16 :: gradu(3,4)
    complex*16 :: kcond
    !
    call getq(q,xx(1),xx(2),xx(3))
    call gradq(gradu,xx(1),xx(2),xx(3))
    !
    p=gm1*(q(5)-0.5d0*(q(2)**2+q(3)**2+q(4)**2)/q(1))
    T=gamma*p/q(1)
    !
    mu=(c2b+1d0)*T*sqrt(T)/(c2b+T) !< sutherland's law for viscosity
    kcond=mu*prfac+turmu*prtrfac   !< heat conduction coefficient
    mu=mu+turmu
    !
    vflux(1)=dcmplx(0d0,0d0)
    vflux(2)=mu*(gradu(1,idir)+gradu(idir,1))
    vflux(3)=mu*(gradu(2,idir)+gradu(idir,2))
    vflux(4)=mu*(gradu(3,idir)+gradu(idir,3))
    vflux(idir+1)=vflux(idir+1)-TWOTHIRD*mu*(gradu(1,1)+gradu(2,2)+gradu(3,3))
    vflux(5)=(q(2)*vflux(2)+q(3)*vflux(3)+q(4)*vflux(4))/q(1)&
         +kcond*gradu(idir,4)
    !
    vflux=reyinv*vflux
    !
  end subroutine viscousFlux
  !
  subroutine gradient(x,y,z,gradu)
    !
    implicit none
    !
    complex*16, intent(in) :: x,y,z
    real*8, intent(out) :: gradu(3,4)
    complex*16 :: xx(3)
    complex*16 :: qq(5)
    complex*16 :: u,v,w,T
    integer :: idir
    !
    do idir=1,3
       !
       xx(1)=x
       xx(2)=y
       xx(3)=z
       xx(idir)=xx(idir)+dcmplx(0d0,eps)
       !
       call getq(qq,xx(1),xx(2),xx(3))
       call getprim(qq,u,v,w,T)
       !
       gradu(idir,1)=imag(u)/eps
       gradu(idir,2)=imag(v)/eps
       gradu(idir,3)=imag(w)/eps
       gradu(idir,4)=imag(T)/eps
       !
    enddo
    !
  end subroutine gradient
  !
  subroutine getprim(q,u,v,w,T)
    !
    implicit none
    !
    complex*16, intent(in) :: q(5)
    complex*16, intent(out) :: u,v,w,T
    !
    u=q(2)/q(1)
    v=q(3)/q(1)
    w=q(4)/q(1)
    T=gamma*(gamma-1)*(q(5)/q(1)-0.5d0*(u**2+v**2+w**2))
    !
    return
  end subroutine getprim
  !
  subroutine set_mms_params(gamma1,pr1,prtr1,rey1)
    !
    implicit none
    !
    real*8, intent(in) :: gamma1,pr1,prtr1,rey1
    !
    gamma=gamma1
    pr=pr1
    prtr=prtr1
    rey=rey1
    !
    gm1=gamma-1
    ggm1=gamma*gm1
    prfac=1d0/(gm1*pr)
    prtrfac=1d0/(gm1*prtr)
    reyinv=1.d0/rey
    TWOTHIRD=2.d0/3.d0
    turmu=0d0
    !
  end subroutine set_mms_params
  !
  subroutine inviscidDivergence(x,s,nq,npts,istor)
    !
    implicit none
    !
    integer, intent(in) :: nq,npts
    real*8, intent(in) :: x(3*npts)
    real*8, intent(inout) :: s(nq*npts)
    character*(*), intent(in) :: istor
    !
    complex*16 :: xp(3),xx(3),flux(5)
    integer :: xstride,qstride,xmult,qmult
    integer :: i,ip,idir,n,iq
    !
    if (istor=='row') then
       xstride=1
       xmult=3
       qstride=1
       qmult=nq
    else
       xstride=npts
       qstride=npts
       xmult=1
       qmult=1
    endif
    !
    do i=1,npts
       !
       ip=(i-1)*xmult+1
       xx(1)=dcmplx(x(ip),0d0)
       ip=ip+xstride
       xx(2)=dcmplx(x(ip),0d0)
       ip=ip+xstride
       xx(3)=dcmplx(x(ip),0d0)
       !
       do idir=1,3
          iq=(i-1)*qmult+1
          xp=xx
          xp(idir)=xp(idir)+dcmplx(0d0,eps)
          call inviscidFlux(idir,flux,xp)
          do n=1,5
             s(iq)=s(iq)-imag(flux(n))/eps
             iq=iq+qstride
          enddo
       enddo
       !        
       !
    enddo
    return
  end subroutine inviscidDivergence
  !
  subroutine viscousDivergence(x,s,nq,npts,istor)
    !
    implicit none
    !
    integer, intent(in) :: nq,npts
    real*8, intent(in) :: x(3*npts)
    real*8, intent(inout) :: s(nq*npts)
    character*(*), intent(in) :: istor
    !
    complex*16 :: xx(3),xp(3),flux(5)
    integer :: i,ip,idir,n,iq
    integer :: xstride,qstride,xmult,qmult
    integer :: idebug
    !
    if (istor=='row') then
       xstride=1
       xmult=3
       qstride=1
       qmult=nq
    else
       xstride=npts
       qstride=npts        
       xmult=1
       qmult=1
    endif
    !
    do i=1,npts
       idebug=0
       !if (i==445) idebug=1
       !
       ip=(i-1)*xmult+1
       xx(1)=dcmplx(x(ip),0d0)
       ip=ip+xstride
       xx(2)=dcmplx(x(ip),0d0)
       ip=ip+xstride
       xx(3)=dcmplx(x(ip),0d0)
       !
       do idir=1,3
          iq=(i-1)*qmult+1
          xp=xx
          xp(idir)=xp(idir)+dcmplx(0d0,eps)
          call viscousFlux(idir,flux,xp,idebug)
          do n=1,5
             s(iq)=s(iq)+imag(flux(n))/eps
             iq=iq+qstride
          enddo
       enddo
       !
    enddo
    return
  end subroutine viscousDivergence
end module NS_mms
!
subroutine get_qvar(q,x,y,z)
  use NS_mms
  real*8, intent(in) :: x,y,z
  real*8, intent(out) :: q(5)
  call getqreal(q,x,y,z)
end subroutine get_qvar
!
subroutine set_params_mms(gamma1,pr1,prtr1,rey1)
  use NS_mms
  real*8, intent(in) :: gamma1,pr1,prtr1,rey1
  call set_mms_params(gamma1,pr1,prtr1,rey1)
end subroutine set_params_mms
!
subroutine inviscidDivergence_mms(x,s,nq,npts)
  use NS_mms
  integer, intent(in) :: nq,npts
  real*8, intent(in) :: x(3*npts)
  real*8, intent(inout) :: s(nq*npts)
  character*7 istor
  istor='column'
  call inviscidDivergence(x,s,nq,npts,istor)
end subroutine inviscidDivergence_mms
!
subroutine viscousDivergence_mms(x,s,nq,npts)
  use NS_mms
  integer, intent(in) :: nq,npts
  real*8, intent(in) :: x(3*npts)
  real*8, intent(inout) :: s(nq*npts)
  character*7 istor
  istor='column'
  call viscousDivergence(x,s,nq,npts,istor)
end subroutine viscousDivergence_mms
