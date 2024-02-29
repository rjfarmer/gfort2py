   

! SPDX-License-Identifier: GPL-2.0+

module gh55

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	contains

	recursive function return_char(x) result(str)
		integer, intent(in) :: x
		character(len=10), dimension((2 ** x - 1) ) :: str

		str = ''
		str(2**x-1) = 'abcdefghil'
	
	end function return_char

end module gh55

        
