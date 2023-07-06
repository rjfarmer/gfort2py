! SPDX-License-Identifier: GPL-2.0+

module elements

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	contains


	elemental integer function ele_func_1(x)
		integer,intent(in) :: x

		ele_func_1 = x*2

	end function ele_func_1

	elemental function ele_func_res(x) result(y)
		integer,intent(in) :: x
		integer :: y

		y = x*2

	end function ele_func_res

	elemental integer function ele_func_2(x, y)
		integer,intent(in) :: x, y

		ele_func_2 = x*2 + y*2

	end function ele_func_2


	elemental subroutine ele_sub_2(x, y)
		real(dp),intent(in) :: x
		real(dp), intent(out) :: y

		y = x*3

	end subroutine ele_sub_2

	subroutine call_ele()

		integer :: x,x1(5),x2(5,5)

		x = 1
		x1 = 2
		x2 = 2

		x = ele_func_1(x)

		x1 = ele_func_1(x1)

		x2 = ele_func_1(x2)


	end subroutine call_ele


end module elements
