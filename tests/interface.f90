! SPDX-License-Identifier: GPL-2.0+

module face

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	interface convert
		module procedure convert_int
		module procedure convert_real
		module procedure convert_real_dp
		module procedure convert_str
		module procedure convert_cmplx
	end interface convert
	

	interface operator(+)
		procedure :: my_add
	end interface 
	
	
	interface operator(-)
		procedure :: my_sub
	end interface 	
	
	
	interface assignment (=)
		procedure :: my_eq
	end interface 
	
	interface operator(.MYUN.)
		procedure :: my_unnary
	end interface 
	
	
	type my_type
		real :: a,b
	end type my_type
	
	
	contains
	
	
	real function my_add(a,b)
		type(my_type), intent(in) :: a
		integer, intent(in) :: b
		
		my_add = a%a*2 +b*2
	end function my_add
	
	real function my_sub(a,b)
		type(my_type), intent(in) :: a
		integer, intent(in) :: b
		
		my_sub = a%a*2 - b*2
	end function my_sub
	
	subroutine my_eq(a, b)
		type(my_type), intent(inout)  :: a
		integer,intent(in) :: b
		
		a%a = b*2
	end subroutine my_eq
	
	logical function my_unnary(a,b)
		type(my_type), intent(in) :: a,b
		
		my_unnary = a%a < b%a
	end function my_unnary


	elemental integer function func_elem_int(x)
		integer,intent(in) :: x

		func_elem_int = x*2

	end function func_elem_int
	
	elemental real function func_elem_real(x)
		real, intent(in) :: x

		func_elem_real = x*2

	end function func_elem_real

	elemental real(dp) function func_elem_real_dp(x)
		real(dp), intent(in) :: x

		func_elem_real_dp = x*2

	end function func_elem_real_dp


	subroutine test(x)
		integer :: x,i
		integer, allocatable,dimension(:) :: xarr,yarr
	
	
		write(*,*) func_elem_int(1)
		write(*,*) func_elem_int((/1,2,3,4,5/))
	
		allocate(xarr(x),yarr(x))
		
		do i=1,x
			xarr(i) = i**2
		end do
		
		yarr = func_elem_int(xarr)
	
	end subroutine test



	integer function convert_int(x)
		integer, intent(in) :: x
		convert_int = x * 5
	end function convert_int

	real function convert_real(x)
		real, intent(in) :: x
		convert_real = x * 5
	end function convert_real
	
	real(dp) function convert_real_dp(x)
		real(dp), intent(in) :: x
		convert_real_dp = x * 5
	end function convert_real_dp
	
	character(len=5) function convert_str(x)
		character(len=1), intent(in) :: x
		character(len=1) :: tmp
		integer :: i
		do i=0,5
			write(tmp,*) x
			convert_str(i:i) = tmp
		end do
	end function convert_str
	
	integer function convert_cmplx(x)
		complex, intent(in) :: x
		convert_cmplx = x * 5
	end function convert_cmplx

end module face
