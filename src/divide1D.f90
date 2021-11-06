!>
!! create 1D partitions and
!! return counts and offsets to the original data
!!
subroutine divide1D(numprocs,n,displs)
  implicit none
  !
  integer, intent(in) :: numprocs
  integer, intent(in) :: n
  integer, intent(out) :: displs(numprocs)
  integer :: nleft,i2,i1,i
  !
  nleft=n
  i2=0
  eloop: do i=0,numprocs-1
     i1=i2+1
     i2=i1+(nleft)/(numprocs-i)-1
     nleft=nleft-(i2-i1+1)
     displs(i+1)=i2
  enddo eloop
end subroutine divide1D
