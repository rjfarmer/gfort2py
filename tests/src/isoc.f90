! SPDX-License-Identifier: GPL-2.0+

module isoc

	use iso_fortran_env, only: output_unit, real128
	use iso_c_binding
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	

	integer, bind(C,name='a_int_bind_c') :: a_int = 1
	
	contains


	integer function func_bind_c(x) bind(C,name='c_bind_func')
		integer, intent(in) :: x
		func_bind_c = 2*x

	end function func_bind_c


end module isoc
