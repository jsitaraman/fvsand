subroutine get_qvar(q,x,y,z)
  use NS_mms
  real*8, intent(in) :: x,y,z
  real*8, intent(out) :: q(5)
  call getqreal(q,x,y,z)
end subroutine get_qvar

subroutine set_params_mms(gamma1,pr1,prtr1,rey1)
  use NS_mms
  real*8, intent(in) :: gamma1,pr1,prtr1,rey1
  call set_mms_params(gamma1,pr1,prtr1,rey1)
end subroutine set_params_mms

subroutine invisidDivergence_mms(x,s,nq,npts)
  use NS_mms
  integer, intent(in) :: nq,npts
  real*8, intent(in) :: x(3*npts)
  real*8, intent(inout) :: s(nq*npts)
  character*7 istor
  istor='column'
  call inviscidDivergence(x,s,nq,npts,istor)
end subroutine invisidDivergence_mms

subroutine viscousDivergence_mms(x,s,nq,npts)
  use NS_mms
  integer, intent(in) :: nq,npts
  real*8, intent(in) :: x(3*npts)
  real*8, intent(inout) :: s(nq*npts)
  character*7 istor
  istor='column'
  call viscousDivergence(x,s,nq,npts,istor)
end subroutine viscousDivergence_mms

