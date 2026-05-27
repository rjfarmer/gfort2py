! SPDX-License-Identifier: GPL-2.0+

module isoc

	use iso_fortran_env, only: output_unit, real128
	use iso_c_binding, only: c_int, c_ptr, c_funptr, c_loc, c_null_ptr, c_f_pointer, c_f_procpointer
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	

	integer, bind(C,name='a_int_bind_c') :: a_int = 1
	integer(c_int), target, save :: c_explicit_ret_arr(3) = (/0_c_int, 0_c_int, 0_c_int/)
	integer(c_int), target, save :: c_pointer_ret_arr(4) = (/0_c_int, 0_c_int, 0_c_int, 0_c_int/)
	integer(c_int), allocatable, target, save :: c_alloc_ret_arr(:)

	abstract interface
		integer(c_int) function c_int_unary_fn(x) bind(C)
			import :: c_int
			integer(c_int), value :: x
		end function c_int_unary_fn
	end interface
	
	contains


	integer function func_bind_c(x) bind(C,name='c_bind_func')
		integer, intent(in) :: x
		func_bind_c = 2*x

	end function func_bind_c

	integer(c_int) function func_bind_c_scalar_in(x) bind(C,name='c_bind_scalar_in')
		integer(c_int), value :: x

		func_bind_c_scalar_in = 3_c_int * x

	end function func_bind_c_scalar_in

	integer(c_int) function func_bind_c_scalar_return() bind(C,name='c_bind_scalar_return')
		func_bind_c_scalar_return = 77_c_int

	end function func_bind_c_scalar_return

	subroutine sub_bind_c_scalar_inout(x) bind(C,name='c_bind_scalar_inout')
		integer(c_int), intent(inout) :: x

		x = x + 11_c_int

	end subroutine sub_bind_c_scalar_inout

	subroutine sub_bind_c_arr_explicit_inout(x) bind(C,name='c_bind_arr_explicit_inout')
		integer(c_int), intent(inout) :: x(3)

		x = x + 1_c_int

	end subroutine sub_bind_c_arr_explicit_inout

	subroutine sub_bind_c_arr_assumed_size_inout(x,n) bind(C,name='c_bind_arr_assumed_size_inout')
		integer(c_int), intent(inout) :: x(*)
		integer(c_int), value :: n
		integer :: i

		do i=1,n
			x(i) = x(i) + 2_c_int
		end do

	end subroutine sub_bind_c_arr_assumed_size_inout

	subroutine sub_bind_c_arr_assumed_shape_inout(x_ptr,n) bind(C,name='c_bind_arr_assumed_shape_inout')
		type(c_ptr), value :: x_ptr
		integer(c_int), value :: n
		integer(c_int), pointer :: x(:)

		call c_f_pointer(x_ptr, x, (/n/))
		x = x + 3_c_int

	end subroutine sub_bind_c_arr_assumed_shape_inout

	function func_bind_c_arr_explicit_ptr_return() result(res) bind(C,name='c_bind_arr_explicit_ptr_return')
		type(c_ptr) :: res

		c_explicit_ret_arr = (/101_c_int, 102_c_int, 103_c_int/)
		res = c_loc(c_explicit_ret_arr)

	end function func_bind_c_arr_explicit_ptr_return

	function func_bind_c_arr_pointer_return() result(res) bind(C,name='c_bind_arr_pointer_return')
		type(c_ptr) :: res

		c_pointer_ret_arr = (/201_c_int, 202_c_int, 203_c_int, 204_c_int/)
		res = c_loc(c_pointer_ret_arr)

	end function func_bind_c_arr_pointer_return

	subroutine sub_bind_c_arr_allocatable_fill(n) bind(C,name='c_bind_arr_allocatable_fill')
		integer(c_int), value :: n
		integer :: i

		if (allocated(c_alloc_ret_arr)) deallocate(c_alloc_ret_arr)
		allocate(c_alloc_ret_arr(n))

		do i=1,n
			c_alloc_ret_arr(i) = 300_c_int + i
		end do

	end subroutine sub_bind_c_arr_allocatable_fill

	function func_bind_c_arr_allocatable_ptr_return(n) result(res) bind(C,name='c_bind_arr_allocatable_ptr_return')
		integer(c_int), value :: n
		type(c_ptr) :: res

		call sub_bind_c_arr_allocatable_fill(n)

		if (allocated(c_alloc_ret_arr)) then
			res = c_loc(c_alloc_ret_arr(1))
		else
			res = c_null_ptr
		end if

	end function func_bind_c_arr_allocatable_ptr_return

	subroutine sub_bind_c_arr_allocatable_inout(x_ptr,n) bind(C,name='c_bind_arr_allocatable_inout')
		type(c_ptr), value :: x_ptr
		integer(c_int), value :: n
		integer(c_int), pointer :: x(:)
		integer(c_int), allocatable :: work(:)

		call c_f_pointer(x_ptr, x, (/n/))
		allocate(work(n))
		work = x + 5_c_int
		x = work
		deallocate(work)

	end subroutine sub_bind_c_arr_allocatable_inout

	subroutine sub_bind_c_proc_arg(fn_ptr, x, res) bind(C,name='c_bind_proc_arg')
		type(c_funptr), value :: fn_ptr
		integer(c_int), value :: x
		integer(c_int), intent(out) :: res
		procedure(c_int_unary_fn), pointer :: fn

		call c_f_procpointer(fn_ptr, fn)
		res = fn(x)

	end subroutine sub_bind_c_proc_arg

	integer(c_int) function func_bind_c_proc_arg(fn_ptr, x) bind(C,name='c_bind_proc_arg_func')
		type(c_funptr), value :: fn_ptr
		integer(c_int), value :: x
		procedure(c_int_unary_fn), pointer :: fn

		call c_f_procpointer(fn_ptr, fn)
		func_bind_c_proc_arg = fn(x)

	end function func_bind_c_proc_arg


end module isoc
