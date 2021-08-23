module boundary_interface
  integer :: npatch
  integer :: ntrifaces,nquadfaces
  integer, allocatable :: faceInfo(:)
  integer, allocatable :: cell2cell(:)

  integer, dimension(3,4)  :: tetf
  integer, dimension(4,5)  :: pyrf
  integer, dimension(4,5)  :: prizmf
  integer, dimension(4,6)  :: hexf
  integer, dimension(6,4)  :: numverts
  integer, dimension(4)    :: iface
  integer :: tetmap(4)
  integer :: pyrmap(5)
  integer :: prizmap(5)
  integer :: hexmap(6)
  !
  data tetf /1,2,3,1,4,2,2,4,3,1,3,4/                        ! ugrid/mcell face connectivity (tet)
  !data tetmap /4,1,2,3/                                     ! map to exodus face ordering
  data tetmap /1,2,3,4/
  data pyrf /1,2,3,4,1,5,2,2,2,5,3,3,4,3,5,5,1,4,5,5/        ! ugrid/mcell face connectivity (pyramid)
  !data pyrmap /5,1,2,3,4/                                     ! map to exodus face ordering
  data pyrmap /1,2,3,4,5/
  data prizmf /1,2,3,3,1,4,5,2,2,5,6,3,1,3,6,4,4,6,5,5/       ! prizm connectivity
  !data prizmap /4,1,2,3,5/                                    ! map to exodus face ordering
  data prizmap /1,2,3,4,5/
  data hexf /1,2,3,4,1,5,6,2,2,6,7,3,3,7,8,4,1,4,8,5,5,8,7,6/ ! ugrid/mcell connectivity (hex)
  data hexmap /1,2,3,4,5,6/
  !data hexmap /5,1,2,3,4,6/                                   ! map to exodus face ordering
  data iface/4,5,5,6/
  data numverts/3,3,3,3,0,0,&
       4,3,3,3,3,0,&
       3,4,4,4,3,0,&
       4,4,4,4,4,4/  
contains
  subroutine deletedata
    if (allocated(faceInfo)) deallocate(faceInfo)
    if (allocated(cell2cell)) deallocate(cell2cell)
  end subroutine deletedata
  !>
  !> find boundary faces in a given graph, using the information of 
  !> boundary nodes that are provided in iflag. 
  !> If all nodes are set to 1, then the
  !> routine will find all the faces of a given graph.
  !>
  !> include 'findboundaryface.h' for the explicit interface
  !> since this subroutine allocates the data it finds
  !> 
  !> Get the connectivity of the faces and also
  !> [cellId,facenum] pair for each face
  !>
  !> Jay Sitaraman
  !> 12/15/2017
  !>
  subroutine findnewboundaryface(cell2cell,faceInfo,iflag,ntrifaces,nquadfaces,ndc4,ndc5,ndc6,ndc8, &
       nnode,ntetra,npyra,nprizm,nhexa)
    !
    implicit none
    !
    integer, allocatable, intent(out) :: cell2cell(:)      !> cell2cell graph
    integer, allocatable, intent(out) :: faceInfo(:)       !> face information (faceconnectivity and cell connections
    integer, intent(inout) :: ntrifaces,nquadfaces         !> number of triangle and quad faces
    integer, intent(in) :: nnode,ntetra,npyra,nprizm,nhexa !> number of nodes and elements in the graph
    integer, intent(in) :: iflag(nnode)                    !> flag to say which one are boundary nodes
    integer, intent(in) :: ndc4(4,ntetra)                  !> tetra graph  
    integer, intent(in) :: ndc5(5,npyra)                   !> pyra graph 
    integer, intent(in) :: ndc6(6,nprizm)                  !> prizm graph  
    integer, intent(in) :: ndc8(8,nhexa)                   !> hex graph  
    !
    ! local variables
    !
    !
    integer :: i,j,k,l,nsum,ncells,maxfaces,nfaces,csum,nbface,icptr
    integer :: flocal(4)
    integer, allocatable :: face(:,:),ipointer(:),iflag_cell(:)
    integer, parameter :: base=1
    real*8 :: t1,t2
    !
    ! find all exposed cell faces
    !
    call cpu_time(t1)
    allocate(iflag_cell(ntetra+npyra+nprizm+nhexa))
    !
    maxfaces=0
    k=0
    do i=1,ntetra
       k=k+1
       iflag_cell(k)=0
       csum=0
       do j=1,4
          csum=csum+iflag(ndc4(j,i)+base)
       enddo
       if (csum > 0 ) then
          iflag_cell(k)=1
          maxfaces=maxfaces+4
       endif
    enddo
    !
    do i=1,npyra
       k=k+1
       iflag_cell(k)=0
       csum=0
       do j=1,5
          csum=csum+iflag(ndc5(j,i)+base)
       enddo
       if (csum > 0 ) then
          iflag_cell(k)=1
          maxfaces=maxfaces+5
       endif
    enddo
    !
    do i=1,nprizm
       k=k+1
       iflag_cell(k)=0
       csum=0
       do j=1,6
          csum=csum+iflag(ndc6(j,i)+base)
       enddo
       if (csum > 0 ) then
          iflag_cell(k)=1
          maxfaces=maxfaces+6
       endif
    enddo
    !
    do i=1,nhexa
       k=k+1
       iflag_cell(k)=0
       csum=0
       do j=1,8
          csum=csum+iflag(ndc8(j,i)+base) 
       enddo
       if (csum >0 ) then
          iflag_cell(k)=1
          maxfaces=maxfaces+8
       endif
    enddo
    !
    ! do a hash insert of the faces now
    !
    allocate(ipointer(nnode))
    allocate(face(9,maxfaces))
    !
    face=0
    ipointer=0
    nfaces=0
    ncells=0
    !
    l=0
    do i=1,ntetra
       l=l+1
       if (iflag_cell(l)==0) cycle
       do j=1,iface(1)
          nsum=0
          flocal=0
          do k=1,numverts(j,1)
             flocal(k)=ndc4(tetf(k,j),i)+base
          enddo
          call insert_face(flocal,face,ipointer,i+ncells,&
               tetmap(j),nfaces,maxfaces,nnode,numverts(j,1))
       enddo
    enddo
    !
    ncells=ncells+ntetra
    !
    do i=1,npyra
       l=l+1
       if (iflag_cell(l)==0) cycle
       do j=1,iface(2)
          nsum=0
          flocal=0
          do k=1,numverts(j,2)
             flocal(k)=ndc5(pyrf(k,j),i)+base
          enddo
          call insert_face(flocal,face,ipointer,i+ncells,&
               pyrmap(j),nfaces,maxfaces,nnode,numverts(j,2))
       enddo
    enddo
    !
    ncells=ncells+npyra
    !
    do i=1,nprizm
       l=l+1
       if (iflag_cell(l)==0) cycle
       do j=1,iface(3)
          nsum=0
          flocal=0
          do k=1,numverts(j,3)
             flocal(k)=ndc6(prizmf(k,j),i)+base
          enddo
          call insert_face(flocal,face,ipointer,i+ncells,&
               prizmap(j),nfaces,maxfaces,nnode,numverts(j,3))
       enddo
    enddo
    !
    ncells=ncells+nprizm
    !
    do i=1,nhexa
       l=l+1
       if (iflag_cell(l)==0) cycle
       do j=1,iface(4)
          nsum=0
          flocal=0
          do k=1,numverts(j,4)
             flocal(k)=ndc8(hexf(k,j),i)+base
          enddo
          call insert_face(flocal,face,ipointer,i+ncells,&
               hexmap(j),nfaces,maxfaces,nnode,numverts(j,4))
       enddo
    enddo
    !
    ! find outer boundary face indices
    !
    ntrifaces=0
    nquadfaces=0
    nbface=0
    !
    allocate(cell2cell(4*ntetra+5*npyra+5*nprizm+6*nhexa))
    allocate(faceInfo(nfaces*8))
    write(6,*) 'ntetra,npyra,nprizm,nhexa,nfaces=',ntetra,npyra,nprizm,nhexa,nfaces
    do i=1,nfaces
       if (face(4,i)==0) then
          faceInfo((i-1)*8+1)=face(3,i)-base
          faceInfo((i-1)*8+2)=face(2,i)-base
          faceInfo((i-1)*8+3)=face(1,i)-base
          faceInfo((i-1)*8+4)=face(1,i)-base
          ntrifaces=ntrifaces+1
       else
          faceInfo((i-1)*8+1)=face(4,i)-base
          faceInfo((i-1)*8+2)=face(3,i)-base
          faceInfo((i-1)*8+3)=face(2,i)-base
          faceInfo((i-1)*8+4)=face(1,i)-base
          nquadfaces=nquadfaces+1
       endif
       faceInfo((i-1)*8+5)=face(5,i)-base
       faceInfo((i-1)*8+6)=face(8,i)-base
       faceInfo((i-1)*8+7)=face(6,i)-base
       faceInfo((i-1)*8+8)=face(9,i)-base       
       cell2cell(cell_index(face(5,i))+face(8,i))=face(6,i)-base
       if (face(6,i) > 0) then
          cell2cell(cell_index(face(6,i))+face(9,i))=face(5,i)-base
       else
          nbface=nbface+1
       endif
    enddo
    !
    write(6,*) 'nbface=',nbface

    ! nbface=0
    ! do i=1,ncells
    !    do j=1,5
    !       if (cell2cell(5*(i-1)+j) .lt. 0) then
    !          nbface=nbface+1
    !       endif
    !    enddo
    ! enddo
    ! write(6,*) 'nbface=',nbface
    !
    deallocate(face)
    deallocate(ipointer)
    deallocate(iflag_cell)
    call cpu_time(t2)
    !
    write(6,*) 'Face calculation time :',t2-t1
    return
  contains
    function cell_index(cin) result (indx)
      integer, intent(in) :: cin
      integer :: indx
      if (cin <= ntetra) then
         indx=4*(cin-1)
      elseif (cin <= ntetra+npyra) then
         indx=5*(cin-(ntetra)-1)+4*ntetra
      elseif (cin <= ntetra+npyra+nprizm) then
         indx=5*(cin-(ntetra+npyra)-1)+4*ntetra+5*npyra
      elseif (cin <= ntetra+npyra+nprizm+nhexa) then
         indx=6*(cin-(ntetra+npyra+nprizm)-1)+4*ntetra+5*npyra+5*nprizm
      endif
    end function cell_index    
    !
  end subroutine findnewboundaryface
  !
  !===========================================================================
  subroutine sort_face(a,n)
    implicit none
    integer :: n
    integer :: a(n)
    integer i,j,tmp

    do i=1,n
       do j=i+1,n
          if (a(j) < a(i)) then
             tmp=a(j)
             a(j)=a(i)
             a(i)=tmp
          endif
       enddo
    enddo

    return
  end subroutine sort_face
  !
  ! insert the face into a hashed list
  !
  !===========================================================================
  subroutine insert_face(flocal,face,ipointer,cellIndex,faceIndex, &
       nfaces,maxfaces,nnode,numpts)
    implicit none
    integer :: nfaces,maxfaces,nnode,numpts
    integer :: face(9,maxfaces)
    integer :: cellIndex,faceIndex
    integer :: flocal(4)
    integer :: ipointer(nnode)
    !
    ! local variables
    !
    integer :: ip,ip0
    integer :: f1(4),f2(4)
    !
    !
    f1=flocal
    call sort_face(f1,numpts)
    !
    ip0=ipointer(f1(1))
    ip=ip0
    checkloop: do while(ip > 0)
       f2=face(1:4,ip)
       call sort_face(f2,numpts)
       if (sum(abs(f1-f2))==0) then
          face(6,ip)=cellIndex
          face(9,ip)=faceIndex
          exit checkloop
       endif
       ip=face(7,ip)
    enddo checkloop
    if (ip==0) then
       nfaces=nfaces+1
       face(1:4,nfaces)=flocal
       face(5,nfaces)=cellIndex
       face(6,nfaces)=0
       face(7,nfaces)=ipointer(f1(1))
       face(8,nfaces)=faceIndex
       face(9,nfaces)=0
       ipointer(f1(1))=nfaces
    endif
    return
  end subroutine insert_face

end module boundary_interface


subroutine get_exposed_faces_prizms(ndc6,nprizm)
  use boundary_interface
  implicit none
  integer, intent(in) :: nprizm
  integer, intent(in) :: ndc6(6*nprizm)
  integer, allocatable :: ndc4(:,:),ndc5(:,:),ndc8(:,:),iflag(:)
  integer :: nnode,ntetra,npyra,nhexa
  
  nnode=maxval(ndc6)+1
  allocate(iflag(nnode))
  iflag=1
  ntetra=0
  npyra=0
  nhexa=0
  
  call findnewboundaryface(cell2cell,faceInfo,iflag,ntrifaces,nquadfaces,&
                           ndc4,ndc5,ndc6,ndc8, &
                           nnode,ntetra,npyra,nprizm,nhexa)
  deallocate(iflag)

end subroutine get_exposed_faces_prizms
  
  
subroutine get_face_count(ntriout,nquadout)
  use boundary_interface
  implicit none
  integer, intent(inout) :: ntriout,nquadout
  ntriout=ntrifaces
  nquadout=nquadfaces
end subroutine get_face_count

subroutine get_graph(cellout,faceout,csize,fsize)
  use boundary_interface
  implicit none
  integer, intent (in) :: fsize,csize
  integer, intent (out):: cellout(csize)
  integer, intent(out) :: faceout(fsize)
  integer :: m,i

  do i=1,fsize
     faceout(i)=faceInfo(i)
  enddo
  do i=1,csize
     cellout(i)=cell2cell(i)
  enddo
     
  call deleteData
  
end subroutine get_graph
