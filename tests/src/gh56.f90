   

! SPDX-License-Identifier: GPL-2.0+

module gh56

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	public :: get_array, print3, my_t

	type my_t
		real, allocatable :: d(:)
		integer :: n
	end type
	
	contains 
	
	function get_array(N) result(retval)
		type(my_t) :: retval
		integer, intent(in) :: N
		integer :: i
		
		allocate( retval%d(N) )
		retval%n = N
		
		! substitute with real data 
		do i=1,N
			retval% d(i) = i
		end do
	end function
	
	subroutine print3(x)
		type(my_t), intent(in) :: x
		print *, x%d(:3)
	end subroutine



end module gh56

        
