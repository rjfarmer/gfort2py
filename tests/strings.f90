! SPDX-License-Identifier: GPL-2.0+

module strings

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(8)
	
	
	character(len=10),parameter :: const_str='1234567890'
	
	character(len=10) :: a_str
	character(len=10) :: a_str_set='abcdefghjk'
	character(:), allocatable :: str_alloc
	
	
	contains
	
	subroutine sub_str_in_explicit(x)
		character(len=10), intent(in) ::x
		write(output_unit,*) trim(x)
	end subroutine sub_str_in_explicit
	
	subroutine sub_str_in_implicit(x)
		character(len=*), intent(in) ::x
		write(output_unit,*) trim(x)
	end subroutine sub_str_in_implicit
	
	subroutine sub_str_multi(x,y,z)
		integer, intent(in) ::x,z
		character(len=*), intent(in) ::y
		write(output_unit,'(I1,1X,A)') x+z,trim(y)
	end subroutine sub_str_multi
	
	subroutine sub_str_alloc(x)
		character(:), allocatable, intent(out) :: x
		x = 'abcdef'
	end subroutine sub_str_alloc
	
	subroutine sub_str_p(zzz)
		character(len=*),pointer, intent(inout) :: zzz
		
		write(output_unit,'(A)') zzz
		
		zzz = 'xyzxyz'
	end subroutine sub_str_p
      

end module strings
