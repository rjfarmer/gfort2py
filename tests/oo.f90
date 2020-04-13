! SPDX-License-Identifier: GPL-2.0+

module oo

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(8)
	
	
	TYPE s_proc
		integer :: a_int
		
		contains
		
		procedure, nopass :: proc_no_pass => sub_dt_no_pass
		procedure, pass(this) :: proc_pass => sub_dt_pass
	
	end type s_proc
	
	TYPE(s_proc) :: p_proc
	
	contains
	
	subroutine sub_dt_no_pass(x)
		integer :: x
		
		write(*,*) 5*x
	
	end subroutine sub_dt_no_pass
	
	
	subroutine sub_dt_pass(this,x)
		class(s_proc), intent(inout) :: this
		integer :: x
		
		this%a_int = 5*x
		
	end subroutine sub_dt_pass
      


end module oo
