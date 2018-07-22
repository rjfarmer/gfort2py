module test2
	implicit none
      integer, parameter :: lp2 = selected_int_kind(8)

   INTEGER :: test2_x

   TYPE s_simple2
      integer           :: x,y,z
   END TYPE s_simple2
   
   TYPE s_struct_AA
      integer           :: a_int
      real              :: a_real
      double precision          :: a_real_dp
      character(len=1_lp2) :: a_str   
      integer,dimension(:,:,:),allocatable :: arr
      integer           :: a_int2
   END TYPE s_struct_AA

end module test2
