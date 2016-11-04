module tester
	implicit none
      integer, parameter :: dp = selected_real_kind(p=15)
      integer, parameter :: dp2=dp+1
      real,parameter :: real_param=9
      integer :: xp=1
      
      real(dp) :: y
      integer :: yyint
      real(dp) :: aaa(4)=(/1.0,2.0,3.0,4.0/)
      
      real(dp),allocatable,dimension(:) :: alloc_1
       real(dp),allocatable,dimension(:,:) :: alloc_2     
 
      real(dp),pointer,dimension(:) :: p_1
       real(dp),pointer,dimension(:,:) :: p_2  
       
      real(dp),dimension(10),target :: t_1         
      
      type struct_tester
         integer :: st1
      end type struct_tester
      
      type EoS_General_Info
         logical :: include_radiation, always_skip_elec_pos, always_include_elec_pos
         integer :: partials_consistency_level
         real(dp) :: mass_fraction_limit_for_PC ! skip any species with abundance < this
         real(dp) :: logRho1_OPAL_SCVH_limit ! don't use OPAL_SCVH for logRho > this
         real(dp) :: logRho2_OPAL_SCVH_limit ! full OPAL_SCVH okay for logRho < this
         real(dp) :: logRho1_PC_limit ! okay for pure PC for logRho > this
         real(dp) :: logRho2_PC_limit ! don't use PC for logRho < this (>= 2.8)
         ! transition log_Gamma for PC to HELM
         real(dp) :: log_Gamma_all_HELM ! HELM for log_Gamma <= this
         real(dp) :: Gamma_all_HELM ! 10**log_Gamma_all_HELM
         real(dp) :: log_Gamma_all_PC ! PC for log_Gamma >= this
         real(dp) :: PC_min_Z ! don't use PC for Z < this
         ! transition Z for OPAL to HELM
         real(dp) :: Z_all_HELM ! HELM for Z >= this
         ! transition temperature zone for OPAL to HELM at high T
         real(dp) :: logT_all_HELM ! HELM for lgT >= this
         real(dp) :: logT_all_OPAL ! OPAL for lgT <= this
         ! transition temperature zone for SCVH to HELM at very low T
         real(dp) :: logT_low_all_HELM ! HELM for lgT <= this
         real(dp) :: logT_low_all_SCVH ! SCVH for lgT >= this
         ! transition energy zone for OPAL to HELM (for eosDE)
         real(dp) :: logE_all_HELM ! HELM for lgE >= this
         real(dp) :: logE_all_OPAL ! OPAL for lgE <= this
         ! transition from HELM fully ionized to HELM fully neutral
         real(dp) :: logT_ion, logT_neutral, max_logRho_neutral
         ! bookkeeping
         integer :: handle
         logical :: in_use
         real(dp),dimension(10) :: xxxx
         real(dp),allocatable,dimension(:,:) :: xxxx2
         type(struct_tester) :: s1
         type(struct_tester),dimension(5) :: s2
         type(struct_tester),allocatable,dimension(:) :: s3
      end type EoS_General_Info
      
      type(eos_general_info) :: seos1,seos2
      type(eos_general_info),allocatable,dimension(:) :: seos1a
      
      
      contains
      
	real(dp) function test_struct(x)
		type(eos_general_info), intent(in) :: x
      test_struct=1.d0
      write(*,*) aaa
      write(*,*) x%include_radiation
   end function test_struct
	
	real(dp) function loglog(x,z)
		real(dp), intent(in) :: x
      integer, intent(in) :: z
      write(*,*) "x= ",x
      write(*,*) "z= ",z,x*real_param
      y=1.d0
      loglog=log10(x)
   end function loglog
   
	subroutine loglogs(x,z)
		real(dp), intent(in) :: x
      integer, intent(in) :: z
      write(*,*) "x= ",x
      write(*,*) "z= ",z
      y=1.d0
      write(*,*) log10(x)
   end subroutine loglogs
 
   subroutine array_in(x)
!      integer,dimension(1:9),intent(in) :: x
      integer,dimension(:),intent(in) :: x
      
      write(*,*) shape(x),"*",ubound(x),"*",lbound(x),"*",size(x)
      write(*,*) x(:)
      write(*,*) "#",loc(x)
      write(*,*)
      
   end subroutine array_in
   
   subroutine array_in_fixed(x)
      integer,dimension(1:9),intent(in) :: x
!      integer,dimension(:),intent(in) :: x
      
      write(*,*) shape(x),"*",ubound(x),"*",lbound(x),"*",size(x)
      write(*,*) x(:)
      write(*,*) "#",loc(x)
      
       
   end subroutine array_in_fixed
   
   subroutine array_in_fixed_2d(x)
      integer,dimension(1:2,1:9),intent(in) :: x
!      integer,dimension(:),intent(in) :: x
      
      write(*,*) shape(x),"*",ubound(x),"*",lbound(x),"*",size(x)
      write(*,*) x(1,:)
      write(*,*) x(2,:)
      write(*,*) "#",loc(x)
      
       
   end subroutine array_in_fixed_2d

   subroutine array_out(x)
!      integer,dimension(1:9),intent(in) :: x
      integer,dimension(:),intent(inout) :: x
      
      write(*,*) shape(x),"*",ubound(x),"*",lbound(x),"*",size(x)
      write(*,*) x(:)
           write(*,*) 
      x=5
      
   end subroutine array_out
   
   
   subroutine array_alloc(x)
!      integer,dimension(1:9),intent(in) :: x
      real,dimension(:),allocatable,intent(out) :: x
      
     allocate(x(1:10))
      x=9.d0
      
   end subroutine array_alloc
   
   subroutine array_alloc2d(x)
!      integer,dimension(1:9),intent(in) :: x
      real,dimension(:,:),allocatable,intent(out) :: x
      
     allocate(x(1:5,1:10))
      x=9.d0
           write(*,*) 
   end subroutine array_alloc2d
   
   subroutine array_alloc3d(x)
!      integer,dimension(1:9),intent(in) :: x
      real,dimension(:,:,:),allocatable,intent(out) :: x
      
     allocate(x(1:5,1:6,7))
      x=9.d0
          write(*,*)  
   end subroutine array_alloc3d

   
   subroutine my_func(in_str,yy,zz,aa,bb,cc,dd,ee,ff,xx,xx2)
      character(len=*),intent(in),optional :: in_str
      integer,intent(in),optional :: xx
      integer,intent(in),pointer :: xx2
      integer,intent(out) :: yy
      real(dp),intent(inout) :: zz
      real(dp),pointer,dimension(:) :: aa
      real(dp),dimension(:) :: bb
      real(dp),dimension(:,:) :: cc
      real(dp),dimension(:),intent(in) :: dd
      real(dp),dimension(:,:),intent(out) :: ee   
      real(dp),dimension(1:10),intent(in) :: ff 
      
      if(present(in_str))then
         write(*,*) "Fortran: ", trim(in_str)
      else
         write(*,*)
      end if

   end subroutine my_func
   
   integer function test_func(x,z)
      integer,intent(in) :: x
      real(dp) :: z
      real(dp) :: y
      
      test_func=x*y
   end function test_func
   
   integer function test_func2(x)
      integer,intent(in),pointer :: x
      
      test_func2=1
   end function test_func2
   
   
   integer function test_fun_pass(x,func)
      integer, intent(in) :: x
      integer, external :: func
   
      test_fun_pass=func(x)
   end function test_fun_pass
   
   complex function test_complex(x)
      complex,intent(in) :: x
      
      write(*,*) x
   end function test_complex
 
end module tester
