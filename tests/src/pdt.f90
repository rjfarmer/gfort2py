! SPDX-License-Identifier: GPL-2.0+

module pdt

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: sp = selected_real_kind(p=8)
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	type pdt_def(k,a)
		integer, kind :: k = sp 
		integer, len :: a
		real(k) :: array(a,a)
	
	end type pdt_def
	
	
	type (pdt_def(dp,3)) :: pdt_dp_3
	type (pdt_def(sp,3)) :: pdt_sp_3
	
	
	contains

	subroutine sub_pdt_dp_3_in(x)
		type (pdt_def(dp,3)), intent(in) :: x

		write(output_unit,*) x%array(1,1), x%array(3,3)
	
	end subroutine sub_pdt_dp_3_in

	subroutine sub_pdt_dp_3_out(x)
		type (pdt_def(dp,3)), intent(out) :: x

		x%array = 0.0_dp
		x%array(1,1) = 101.0_dp
		x%array(2,2) = 202.0_dp
		x%array(3,3) = 303.0_dp
	
	end subroutine sub_pdt_dp_3_out

	subroutine sub_pdt_dp_3_inout(x)
		type (pdt_def(dp,3)), intent(inout) :: x

		x%array = x%array + 10.0_dp
	
	end subroutine sub_pdt_dp_3_inout

	function func_return_pdt_dp_3() result(res)
		type (pdt_def(dp,3)) :: res

		res%array = reshape((/ &
			11.0_dp, 21.0_dp, 31.0_dp, &
			12.0_dp, 22.0_dp, 32.0_dp, &
			13.0_dp, 23.0_dp, 33.0_dp  &
		/), (/3,3/))
	
	end function func_return_pdt_dp_3


	subroutine sub_write_pdt()
	
		write(output_unit,*) pdt_dp_3%array
	
	end subroutine sub_write_pdt
	
	subroutine sub_pdt()
		type (pdt_def(lp,5)) :: x
		
		x%array = 2
	end subroutine sub_pdt

end module pdt
