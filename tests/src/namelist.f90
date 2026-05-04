! SPDX-License-Identifier: GPL-2.0+

module namelist

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	

	integer :: a_int
	real :: b_real
	real(dp) :: b_real_dp
	character(len=100) :: c_str

	namelist /namelist1/ a_int, b_real, b_real_dp, c_str

	
	contains



end module namelist
 