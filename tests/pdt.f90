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


	subroutine sub_write_pdt()
	
		write(*,*) pdt_dp_3%array
	
	end subroutine sub_write_pdt
	
	subroutine sub_pdt()
		type (pdt_def(lp,5)) :: x
		
		x%array = 2
	end subroutine sub_pdt

end module pdt
