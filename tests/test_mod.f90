module tester
   use iso_fortran_env, only: output_unit

	implicit none
      
      ! Parameters
      integer, parameter :: dp = selected_real_kind(p=15)
      integer, parameter :: qp = selected_real_kind(p=31)
      integer, parameter :: lp = selected_int_kind(8)
      
      integer, parameter      :: const_int=1
      integer, parameter      :: const_int_p1=const_int+1
      integer(lp), parameter  :: const_int_lp=1_lp
      
      real, parameter     :: const_real=1.0
      real(dp), parameter :: const_real_dp=1.0_dp
      real(qp), parameter :: const_real_qp=1.0_qp
      complex, parameter  :: const_cmplx=(1.0,1.0)
      complex(dp), parameter  :: const_cmplx_dp=(1.0_dp,1.0_dp)
      complex(qp), parameter  :: const_cmplx_qp=(1.0_qp,1.0_qp)
      
      character(len=10),parameter :: const_str='1234567890'
      
      integer,parameter,dimension(10) :: const_int_arr=(/1,2,3,4,5,6,7,8,9,0/)
      real,parameter,dimension(10) :: const_real_arr=(/1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0/)
      real(dp),parameter,dimension(10) :: const_real_dp_arr=(/1_dp,2_dp,3_dp,4_dp,5_dp,6_dp,7_dp,8_dp,9_dp,0_dp/)


      ! Variables
      integer           :: a_int
      integer(lp)       :: a_int_lp
      real              :: a_real
      real(dp)          :: a_real_dp
      real(qp)          :: a_real_qp
      character(len=10) :: a_str
      complex           :: a_cmplx
      complex(dp)       :: a_cmplx_dp
      complex(qp)       :: a_cmplx_qp
      
      integer,pointer            :: a_int_point => null()
      integer(lp),pointer        :: a_int_lp_point => null()
      real,pointer               :: a_real_point => null()
      real(dp),pointer           :: a_real_dp_point => null()
      real(qp),pointer           :: a_real_qp_point => null()
      character(len=10),pointer  :: a_str_point => null()
      
      integer,target            :: a_int_target
      integer(lp),target        :: a_int_lp_target
      real,target               :: a_real_target
      real(dp),target           :: a_real_dp_target
      real(qp),target           :: a_real_qp_target
      character(len=10),target  :: a_str_target
      
      
      ! Arrays
      integer, dimension(5) :: b_int_exp_1d
      integer, dimension(5,5) :: b_int_exp_2d
      integer, dimension(5,5,5) :: b_int_exp_3d
      integer, dimension(5,5,5,5) :: b_int_exp_4d
      integer, dimension(5,5,5,5,5) :: b_int_exp_5d
      
      real, dimension(5) :: b_real_exp_1d
      real, dimension(5,5) :: b_real_exp_2d
      real, dimension(5,5,5) :: b_real_exp_3d
      real, dimension(5,5,5,5) :: b_real_exp_4d
      real, dimension(5,5,5,5,5) :: b_real_exp_5d
      
      real(dp), dimension(5) :: b_real_dp_exp_1d
      real(dp), dimension(5,5) :: b_real_dp_exp_2d
      real(dp), dimension(5,5,5) :: b_real_dp_exp_3d
      real(dp), dimension(5,5,5,5) :: b_real_dp_exp_4d
      real(dp), dimension(5,5,5,5,5) :: b_real_dp_exp_5d
      
      
      integer, allocatable, dimension(:) :: c_int_alloc_1d
      integer, allocatable, dimension(:,:) :: c_int_alloc_2d
      integer, allocatable, dimension(:,:,:) :: c_int_alloc_3d
      integer, allocatable, dimension(:,:,:,:) :: c_int_alloc_4d
      integer, allocatable, dimension(:,:,:,:,:) :: c_int_alloc_5d
      
      real, allocatable, dimension(:) :: c_real_alloc_1d
      real, allocatable, dimension(:,:) :: c_real_alloc_2d
      real, allocatable, dimension(:,:,:) :: c_real_alloc_3d
      real, allocatable, dimension(:,:,:,:) :: c_real_alloc_4d
      real, allocatable, dimension(:,:,:,:,:) :: c_real_alloc_5d
      
      real(dp), allocatable, dimension(:) :: c_real_dp_alloc_1d
      real(dp), allocatable, dimension(:,:) :: c_real_dp_alloc_2d
      real(dp), allocatable, dimension(:,:,:) :: c_real_dp_alloc_3d
      real(dp), allocatable, dimension(:,:,:,:) :: c_real_dp_alloc_4d
      real(dp), allocatable, dimension(:,:,:,:,:) :: c_real_dp_alloc_5d
      

      integer, pointer, dimension(:) :: d_int_point_1d => null()
      integer, pointer, dimension(:,:) :: d_int_point_2d => null()
      integer, pointer, dimension(:,:,:) :: d_int_point_3d => null()
      integer, pointer, dimension(:,:,:,:) :: d_int_point_4d => null()
      integer, pointer, dimension(:,:,:,:,:) :: d_int_point_5d => null()
      
      real, pointer, dimension(:) :: d_real_point_1d => null()
      real, pointer, dimension(:,:) :: d_real_point_2d => null()
      real, pointer, dimension(:,:,:) :: d_real_point_3d => null()
      real, pointer, dimension(:,:,:,:) :: d_real_point_4d => null()
      real, pointer, dimension(:,:,:,:,:) :: d_real_point_5d => null()
      
      real(dp), pointer, dimension(:) :: d_real_dp_point_1d => null()
      real(dp), pointer, dimension(:,:) :: d_real_dp_point_2d => null()
      real(dp), pointer, dimension(:,:,:) :: d_real_dp_point_3d => null()
      real(dp), pointer, dimension(:,:,:,:) :: d_real_dp_point_4d => null()
      real(dp), pointer, dimension(:,:,:,:,:) :: d_real_dp_point_5d => null()
      
      integer, target, dimension(5) :: e_int_target_1d
      integer, target, dimension(5,5) :: e_int_target_2d
      integer, target, dimension(5,5,5) :: e_int_target_3d
      integer, target, dimension(5,5,5,5) :: e_int_target_4d
      integer, target, dimension(5,5,5,5,5) :: e_int_target_5d
      
      real, target, dimension(5) :: e_real_target_1d
      real, target, dimension(5,5) :: e_real_target_2d
      real, target, dimension(5,5,5) :: e_real_target_3d
      real, target, dimension(5,5,5,5) :: e_real_target_4d
      real, target, dimension(5,5,5,5,5) :: e_real_target_5d
      
      real(dp), target, dimension(5) :: e_real_dp_target_1d
      real(dp), target, dimension(5,5) :: e_real_dp_target_2d
      real(dp), target, dimension(5,5,5) :: e_real_dp_target_3d
      real(dp), target, dimension(5,5,5,5) :: e_real_dp_target_4d
      real(dp), target, dimension(5,5,5,5,5) :: e_real_dp_target_5d
      
      
      TYPE s_struct_basic
         integer           :: a_int
         integer(lp)       :: a_int_lp
         real              :: a_real
         real(dp)          :: a_real_dp
         character(len=10) :: a_str    
         integer, dimension(5) :: b_int_exp_1d 
         integer, allocatable, dimension(:) :: c_int_alloc_1d
         integer, pointer, dimension(:) :: d_int_point_1d => null()
      END TYPE s_struct_basic
      
      
      TYPE s_struct_nested
         integer           :: a_int
         TYPE(s_struct_basic) :: f_struct
      END TYPE s_struct_nested
      
      TYPE s_struct_nested_2
         integer           :: a_int
         TYPE(s_struct_nested) :: f_nested
      END TYPE s_struct_nested_2
      
      TYPE(s_struct_basic) :: f_struct
      TYPE(s_struct_basic),dimension(2) :: f_struct_exp_2d
      TYPE(s_struct_basic),dimension(2,2) :: f_struct_exp_1d
      TYPE(s_struct_basic),dimension(:),  allocatable :: f_struct_alloc_1d
      TYPE(s_struct_basic),dimension(:,:),allocatable :: f_struct_alloc_2d
      TYPE(s_struct_basic),dimension(:),  pointer :: f_struct_point_1d => null()
      TYPE(s_struct_basic),dimension(:,:),pointer :: f_struct_point_2d => null()
      TYPE(s_struct_basic),dimension(2),  target :: f_struct_target_1d
      TYPE(s_struct_basic),dimension(2,2),target :: f_struct_target_2d


      TYPE(s_struct_nested) :: g_struct
      TYPE(s_struct_nested),dimension(2) :: g_struct_exp_2d
      TYPE(s_struct_nested),dimension(2,2) :: g_struct_exp_1d
      TYPE(s_struct_nested),dimension(:),  allocatable :: g_struct_alloc_1d
      TYPE(s_struct_nested),dimension(:,:),allocatable :: g_struct_alloc_2d
      TYPE(s_struct_nested),dimension(:),  pointer :: g_struct_point_1d => null()
      TYPE(s_struct_nested),dimension(:,:),pointer :: g_struct_point_2d => null()
      TYPE(s_struct_nested),dimension(2),  target :: g_struct_target_1d
      TYPE(s_struct_nested),dimension(2,2),target :: g_struct_target_2d
      
      
      TYPE(s_struct_nested_2) :: h_struct
      TYPE(s_struct_nested_2),dimension(2) :: h_struct_exp_2d
      TYPE(s_struct_nested_2),dimension(2,2) :: h_struct_exp_1d
      TYPE(s_struct_nested_2),dimension(:),  allocatable :: h_struct_alloc_1d
      TYPE(s_struct_nested_2),dimension(:,:),allocatable :: h_struct_alloc_2d
      TYPE(s_struct_nested_2),dimension(:),  pointer :: h_struct_point_1d => null()
      TYPE(s_struct_nested_2),dimension(:,:),pointer :: h_struct_point_2d => null()
      TYPE(s_struct_nested_2),dimension(2),  target :: h_struct_target_1d
      TYPE(s_struct_nested_2),dimension(2,2),target :: h_struct_target_2d


      contains
      
      
      subroutine sub_no_args()
         write(output_unit,*) "1"
      end subroutine sub_no_args
      
      integer function func_int_no_args()
         func_int_no_args=2
         write(output_unit,*) 2
      end function func_int_no_args
      
      real function func_real_no_args()
         func_real_no_args=3.0
         write(output_unit,*) 3.0
      end function func_real_no_args
      
      real(dp) function func_real_dp_no_args()
         func_real_dp_no_args=4.0_dp
         write(output_unit,*) 4.0_dp
      end function func_real_dp_no_args
      
      subroutine sub_int_in(x)
         integer, intent(in) ::x
         write(output_unit,*) 2*x
      end subroutine sub_int_in
      
      subroutine sub_int_out(x)
         integer, intent(out) :: x
         x=1
      end subroutine sub_int_out
      
      subroutine sub_int_inout(x)
         integer, intent(inout) :: x
         x=2*x
      end subroutine sub_int_inout     
      
      subroutine sub_int_no_intent(x)
         integer :: x
         x=2*x
      end subroutine sub_int_no_intent 
      
      integer function func_int_in(x)
         integer, intent(in) ::x
         func_int_in=2*x
      end function func_int_in
      
      integer function func_int_in_multi(x,y,z)
         integer, intent(in) ::x,y,z
         func_int_in_multi=x+y+z
      end function func_int_in_multi
      
      subroutine sub_alter_mod()
         a_int=99
         a_int_lp=99_lp
         a_real=99.0
         a_real_dp=99.0_dp
         a_real_qp=99.0_qp
         a_str="9999999999"
         a_cmplx=(99.0,99.0)
         a_cmplx_dp=(99.0_dp,99.0_dp)
         a_cmplx_qp=(00.0_qp,99.0_qp)
      end subroutine sub_alter_mod
      
   
      subroutine sub_alloc_int_1d_arrs()
      
         if(.not. allocated(c_int_alloc_1d)) allocate(c_int_alloc_1d(5))
         if(.not. allocated(c_int_alloc_2d)) allocate(c_int_alloc_2d(5,5))
         if(.not. allocated(c_int_alloc_3d)) allocate(c_int_alloc_3d(5,5,5))
         if(.not. allocated(c_int_alloc_4d)) allocate(c_int_alloc_4d(5,5,5,5))
         if(.not. allocated(c_int_alloc_5d)) allocate(c_int_alloc_5d(5,5,5,5,5))
         
         c_int_alloc_1d=1
         c_int_alloc_2d=1
         c_int_alloc_3d=1
         c_int_alloc_4d=1  
         c_int_alloc_5d=1
      
      end subroutine sub_alloc_int_1d_arrs
      
end module tester
