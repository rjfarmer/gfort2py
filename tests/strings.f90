! SPDX-License-Identifier: GPL-2.0+

module strings

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	character(len=10),parameter :: const_str='1234567890'
	
	character(len=10) :: a_str
	character(len=10) :: a_str_set='abcdefghjk'
	character(:), allocatable :: str_alloc


	character(len=10),dimension(5) :: a_str_exp_1d
	character(len=10),dimension(5,5) :: a_str_exp_2d
	character(len=10),dimension(5,5,5) :: a_str_exp_3d
	
	
	character(len=10),dimension(:),allocatable :: b_str_alloc_1d
	character(len=10),dimension(:,:),allocatable :: b_str_alloc_2d
	character(len=10),dimension(:,:,:),allocatable :: b_str_alloc_3d


	character(len=2),dimension(3), parameter :: a_str_p_1d = (/'aa','bb','cc'/)



	type str_array_dt
		integer :: start_guard = 123456789
		character(len=10),dimension(5) :: a_str_exp_1d
		integer :: end_guard = 123456789
	end type str_array_dt

	type str_array_dt_alloc
		integer :: start_guard = 123456789
		character(len=10),dimension(:),allocatable :: b_str_alloc_1d
		integer :: end_guard = 123456789
	end type str_array_dt_alloc


	type str_array_dt_out
		character(len=12) :: a_str1, a_str2
		integer ::  a_int
	end type str_array_dt_out

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
	
	subroutine sub_str_alloc(x_alloc)
		character(:), allocatable, intent(out) :: x_alloc
		x_alloc = 'abcdef'
	end subroutine sub_str_alloc
	
	subroutine sub_str_p(zzz)
		character(len=*),pointer, intent(inout) :: zzz
		
		write(output_unit,'(A)') zzz
		
		zzz = 'xyzxyz'
	end subroutine sub_str_p
      
      
	character(len=5) function func_ret_str(x)
		character(len=5) :: x
		
		func_ret_str = x
		func_ret_str(1:1) = 'A'
		
	end function func_ret_str
	
	
	logical function check_str_alloc(ver)
		integer :: ver
		
		check_str_alloc = .false.
		
		if(allocated(str_alloc)) then
		    !write(*,*) allocated(str_alloc)
			!write(*,*) str_alloc
			if(ver==1) then
				if(str_alloc(1:16)=='abcdefghijklmnop')  check_str_alloc = .true.
			else if (ver==2) then 
				if(str_alloc(1:8)=='12346578')  check_str_alloc = .true.
			end if
		end if
	
	end function check_str_alloc
	

	subroutine alter_str_alloc()
		
		str_alloc = 'qwerty'
	
	end subroutine alter_str_alloc
	

	subroutine sub_str_alloc2(x)
		character(len=:),allocatable,intent(inout) :: x
		
		if(.not.allocated(x)) then
			x='zxcvbnm'
			return
		end if

		if(allocated(x)) then
			write(*,*) "*",x,"*",len_trim(x)
			if(x == 'qwerty') then
				write(*,*) "Set 1"
				x = 'asdfghjkl'
				! write(*,*) "Set 2"
			end if
		end if
	
	end subroutine sub_str_alloc2
	

	
	function func_str_int_len(i) result(s)
		! Github issue #12
		integer, intent(in) :: i
		character(len=str_int_len(i)) :: s
		write(s, '(i0)') i
	end function func_str_int_len
		  
	pure integer function str_int_len(i) result(sz)
		! Returns the length of the string representation of 'i'
		integer, intent(in) :: i
		integer, parameter :: MAX_STR = 100
		character(MAX_STR) :: s
		! If 's' is too short (MAX_STR too small), Fortran will abort with:
		! "Fortran runtime error: End of record"
		write(s, '(i0)') i
		sz = len_trim(s)
	end function str_int_len


	subroutine str_array_inout(x)
		character(len=10),dimension(5) :: x

		integer :: i

		write(*,*) x(1:5)


		x(1) = 'zzzzzzzzzz'
		x(2) = 'yyyyyyyyyy'
		x(3) = 'qqqqqqqqqq'
		x(4) = 'wwwwwwwwww'
		x(5) = 'xxxxxxxxxx'


	end subroutine str_array_inout


	subroutine str_array_inout2(x,n)
		character(len=10),dimension(n) :: x
		integer,intent(in) :: n

		integer :: i

		write(*,*) x(1:5)


		x(1) = 'zzzzzzzzzz'
		x(2) = 'yyyyyyyyyy'
		x(3) = 'qqqqqqqqqq'
		x(4) = 'wwwwwwwwww'
		x(5) = 'xxxxxxxxxx'


	end subroutine str_array_inout2


	subroutine str_array_inout3(x)
		character(len=10),dimension(:) :: x

		integer :: i

		write(*,*) x(1:5)


		x(1) = 'zzzzzzzzzz'
		x(2) = 'yyyyyyyyyy'
		x(3) = 'qqqqqqqqqq'
		x(4) = 'wwwwwwwwww'
		x(5) = 'xxxxxxxxxx'


	end subroutine str_array_inout3


	subroutine str_array_inout4(x)
		character(len=*),dimension(5) :: x

		integer :: i

		write(*,*) x(1:5)


		x(1) = 'zzzzzzzzzz'
		x(2) = 'yyyyyyyyyy'
		x(3) = 'qqqqqqqqqq'
		x(4) = 'wwwwwwwwww'
		x(5) = 'xxxxxxxxxx'


	end subroutine str_array_inout4


	subroutine str_array_allocate(x)
		character(len=10),dimension(:),allocatable :: x

		allocate(x(5))

		x(1) = 'zzzzzzzzzz'
		x(2) = 'yyyyyyyyyy'
		x(3) = 'qqqqqqqqqq'
		x(4) = 'wwwwwwwwww'
		x(5) = 'xxxxxxxxxx'


	end subroutine str_array_allocate


	logical function func_str_array_dt(x)
		type(str_array_dt) :: x

		func_str_array_dt = .false.

		if(x%start_guard/=123456789 .or. x% end_guard/=123456789) then
			return
		end if


		if(x%a_str_exp_1d(1) /= 'zzzzzzzzzz') return
		if(x%a_str_exp_1d(2) /= 'yyyyyyyyyy') return
		if(x%a_str_exp_1d(3) /= 'qqqqqqqqqq') return
		if(x%a_str_exp_1d(4) /= 'wwwwwwwwww') return
		if(x%a_str_exp_1d(5) /= 'xxxxxxxxxx') return

		func_str_array_dt = .true.


	end function func_str_array_dt


	logical function func_str_array_dt_alloc(x)
		type(str_array_dt_alloc) :: x

		func_str_array_dt_alloc = .false.

		if(x%start_guard/=123456789 .or. x% end_guard/=123456789) then
			write(*,*) x%start_guard, x%end_guard
			return
		end if

		allocate(x% b_str_alloc_1d(5))


		x% b_str_alloc_1d(1) = 'zzzzzzzzzz'
		x% b_str_alloc_1d(2) = 'yyyyyyyyyy'
		x% b_str_alloc_1d(3) = 'qqqqqqqqqq'
		x% b_str_alloc_1d(4) = 'wwwwwwwwww'
		x% b_str_alloc_1d(5) = 'xxxxxxxxxx'

		func_str_array_dt_alloc = .true.


	end function func_str_array_dt_alloc


	subroutine check_a_str_exp_1d()


		write(*,*) a_str_exp_1d(1:5)


		a_str_exp_1d(1) = 'zzzzzzzzzz'
		a_str_exp_1d(2) = 'yyyyyyyyyy'
		a_str_exp_1d(3) = 'qqqqqqqqqq'
		a_str_exp_1d(4) = 'wwwwwwwwww'
		a_str_exp_1d(5) = 'xxxxxxxxxx'

	end subroutine check_a_str_exp_1d
		  

	subroutine alloc_b_str_alloc_1d()

		if(allocated(b_str_alloc_1d)) deallocate(b_str_alloc_1d)

		allocate(b_str_alloc_1d(5))

		b_str_alloc_1d(1) = 'zzzzzzzzzz'
		b_str_alloc_1d(2) = 'yyyyyyyyyy'
		b_str_alloc_1d(3) = 'qqqqqqqqqq'
		b_str_alloc_1d(4) = 'wwwwwwwwww'
		b_str_alloc_1d(5) = 'xxxxxxxxxx'		

	end subroutine alloc_b_str_alloc_1d


	logical function check_b_str_alloc_1d()

		check_b_str_alloc_1d  =.false.

		if(.not.allocated(b_str_alloc_1d)) return


		if(b_str_alloc_1d(1) /= 'zzzzzzzzzz') return
		if(b_str_alloc_1d(2) /= 'yyyyyyyyyy') return
		if(b_str_alloc_1d(3) /= 'qqqqqqqqqq') return
		if(b_str_alloc_1d(4) /= 'wwwwwwwwww') return
		if(b_str_alloc_1d(5) /= 'xxxxxxxxxx') return

		check_b_str_alloc_1d = .true.

	end function check_b_str_alloc_1d


	subroutine set_str_array_dt_out(x)
		type(str_array_dt_out),intent(out) :: x

		x%a_str1 = 'qwertyuiop[]'
		x%a_str2 = 'asdfghjkl;zx'

		x% a_int = 99

	end subroutine set_str_array_dt_out


	subroutine set_chr_star_star(x)
		character*(*), intent(out) :: x
		x = "abcdefghijkl"
	end subroutine set_chr_star_star


	subroutine check_assumed_shape_str(x)
		character(len=*), dimension(:), intent(in) :: x
		integer  :: i


		do i=1,ubound(x,dim=1)
			write(*,*) trim(x(i))
		end do


	end subroutine check_assumed_shape_str


	integer function check_str_opt(x, n)
		integer, intent(in) :: n
		character*(N), optional, intent(inout) :: x

		if(present(x)) then
			if(x(1:6) == '123456') then
				check_str_opt = 1
			else
				check_str_opt = 2
			end if
		else
			check_str_opt = 3
		end if

		if(n==4) check_str_opt = 4

	end function check_str_opt



end module strings
