! SPDX-License-Identifier: GPL-2.0+

module ptrs

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	integer,pointer            :: a_int_point => null()
	integer(lp),pointer        :: a_int_lp_point => null()
	real,pointer               :: a_real_point => null()
	real(dp),pointer           :: a_real_dp_point => null()
	character(len=10),pointer  :: a_str_point => null()
	
	integer,target            :: a_int_target
	integer(lp),target        :: a_int_lp_target
	real,target               :: a_real_target
	real(dp),target           :: a_real_dp_target
	character(len=10),target  :: a_str_target
	
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
	
	
	contains
	
	subroutine sub_set_ptrs()
	
		if(associated(d_int_point_1d)) write(*,*) d_int_point_1d
		d_int_point_1d => e_int_target_1d
		
		d_int_point_1d = 9
	
	end subroutine sub_set_ptrs



end module ptrs
