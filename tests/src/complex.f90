! SPDX-License-Identifier: GPL-2.0+

module comp
	use iso_fortran_env, only: output_unit, real128


	implicit none
	
		! Parameters
		integer, parameter :: dp = selected_real_kind(p=15)
		integer, parameter :: qp = selected_real_kind(p=30)
		integer, parameter :: lp = selected_int_kind(16)
		
		
		complex, parameter         :: const_cmplx=(1.0,1.0)
		complex(dp), parameter  :: const_cmplx_dp=(1.0_dp,1.0_dp)
		
		
		complex             :: a_cmplx
		complex(dp)       :: a_cmplx_dp
		
		
		complex              :: a_cmplx_set
		complex(dp)       :: a_cmplx_dp_set

		complex, dimension(5) :: a_cmplx_arr
		complex(dp), dimension(2,3) :: a_cmplx_dp_arr
	
	contains

	subroutine sub_cmplx_inout(c)
		complex, intent(inout) :: c
	
		c =c *5
	
	end subroutine sub_cmplx_inout

	complex function func_ret_cmplx(c)
		complex, intent(inout) :: c
	
		func_ret_cmplx = c*5
		
	end function func_ret_cmplx

	subroutine sub_cmplx_value(c,cc)
		complex, intent(in), value :: c
		complex, intent(out) :: cc
	
		cc = c *5
	
	end subroutine sub_cmplx_value

	logical function func_cmplx_explicit_arr_2d(x) result(res)
		complex, dimension(2,3), intent(inout) :: x
		res = .false.
		if(x(2,1) == (2.0,1.0)) res = .true.
		x(1,1) = (9.0,-1.0)
	end function func_cmplx_explicit_arr_2d

	logical function func_cmplx_assumed_shape_arr_1d(x) result(res)
		complex, dimension(:), intent(inout) :: x
		res = .false.
		if(x(1) == (2.0,1.0)) res = .true.
		x = (5.0,2.0)
	end function func_cmplx_assumed_shape_arr_1d

	logical function func_cmplx_assumed_size_arr_1d(x) result(res)
		complex, intent(inout) :: x(*)
		res = .false.
		if(x(2) == (3.0,-2.0)) res = .true.
		x(1) = (11.0,-7.0)
	end function func_cmplx_assumed_size_arr_1d

	logical function func_cmplx_assumed_rank_arr(x) result(res)
		complex, dimension(..), intent(in) :: x
		res = .false.
		select rank(x)
		rank(1)
			if(size(x) >= 1) res = (x(1) == (2.0,1.0))
		rank(2)
			if(size(x,1) >= 2 .and. size(x,2) >= 1) res = (x(2,1) == (2.0,1.0))
		rank default
			res = .false.
		end select
	end function func_cmplx_assumed_rank_arr

	function func_ret_cmplx_arr_1d() result(v)
		complex, dimension(3) :: v
		integer :: i

		do i=1,3
			v(i) = cmplx(real(i), -real(i))
		end do
	end function func_ret_cmplx_arr_1d

	function func_ret_cmplx_arr_n(n) result(v)
		integer, intent(in) :: n
		complex, dimension(n) :: v
		integer :: i

		do i=1,n
			v(i) = cmplx(real(i), real(i+1))
		end do
	end function func_ret_cmplx_arr_n


end module comp
