! SPDX-License-Identifier: GPL-2.0+

module quad
	use iso_fortran_env, only: output_unit, real128

	implicit none
	
	     ! Parameters
		integer, parameter :: dp = selected_real_kind(p=15)
		integer, parameter :: qp = selected_real_kind(p=30)
		integer, parameter :: lp = selected_int_kind(16)
		
#ifdef __GFC_REAL_16__
    
		real(qp), parameter :: const_real_qp=1.0_qp

		real(qp)          :: a_real_qp
		real(qp), dimension(4) :: a_real_qp_arr = [1.0_qp, 2.0_qp, 3.0_qp, 4.0_qp]
		real(qp), allocatable, dimension(:) :: a_real_qp_alloc_arr
		
		! Set variables
		real(qp)          :: a_real_qp_set=9.0_qp

		complex(qp), parameter  :: const_cmplx_qp=(1.0_qp,1.0_qp)
		complex(qp)       :: a_cmplx_qp
		complex(qp)       :: a_cmplx_qp_set
		complex(qp), dimension(4) :: a_cmplx_qp_arr = (/ &
			cmplx(1.0_qp, -1.0_qp, kind=qp), &
			cmplx(2.0_qp, -2.0_qp, kind=qp), &
			cmplx(3.0_qp, -3.0_qp, kind=qp), &
			cmplx(4.0_qp, -4.0_qp, kind=qp) /)
		complex(qp), allocatable, dimension(:) :: a_cmplx_qp_alloc_arr

        real(qp),pointer           :: a_real_qp_point => null()
        
        real(qp),target           :: a_real_qp_target

		
	contains
		
		subroutine sub_alter_mod()
			a_real_qp=99.0_qp
		end subroutine sub_alter_mod
      
		logical function func_check_mod()
			func_check_mod = .false.
		
			if(a_real_qp==5.0_qp) then
			    
			    func_check_mod = .true.
			end if

		end function func_check_mod
      		
		subroutine sub_test_quad(y,x)
			real(qp), intent(in) :: y
			real(qp), intent(out) :: x
		
			x = y * 3
		
		end subroutine sub_test_quad


		subroutine sub_qp_scalar_inout(x)
			real(qp), intent(inout) :: x

			x = x * 2

		end subroutine sub_qp_scalar_inout


		subroutine sub_qcmplx_qp_scalar_inout(x)
			complex(qp), intent(inout) :: x

			x = x + (1.0_qp, -1.0_qp)

		end subroutine sub_qcmplx_qp_scalar_inout


		subroutine sub_alloc_qp_module_arr(n, val)
			integer, intent(in) :: n
			real(qp), intent(in) :: val

			if (allocated(a_real_qp_alloc_arr)) deallocate(a_real_qp_alloc_arr)
			allocate(a_real_qp_alloc_arr(n))
			a_real_qp_alloc_arr = val

		end subroutine sub_alloc_qp_module_arr


		subroutine sub_alloc_qcmplx_qp_module_arr(n, val)
			integer, intent(in) :: n
			complex(qp), intent(in) :: val

			if (allocated(a_cmplx_qp_alloc_arr)) deallocate(a_cmplx_qp_alloc_arr)
			allocate(a_cmplx_qp_alloc_arr(n))
			a_cmplx_qp_alloc_arr = val

		end subroutine sub_alloc_qcmplx_qp_module_arr


		logical function func_qp_explicit_arr_1d(x) result(res)
			real(qp), dimension(4), intent(inout) :: x

			res = .false.
			if (x(2) == 2.0_qp) res = .true.
			x = x + 1.0_qp

		end function func_qp_explicit_arr_1d


		logical function func_qp_assumed_shape_arr_1d(x) result(res)
			real(qp), dimension(:), intent(inout) :: x

			res = .false.
			if (size(x) >= 2) res = (x(2) == 2.0_qp)
			x = x + 2.0_qp

		end function func_qp_assumed_shape_arr_1d


		logical function func_qp_assumed_size_arr_1d(x, n) result(res)
			real(qp), intent(inout) :: x(*)
			integer, intent(in) :: n

			res = .false.
			if (n >= 2) res = (x(2) == 2.0_qp)
			x(1:n) = x(1:n) + 3.0_qp

		end function func_qp_assumed_size_arr_1d


		logical function func_qp_assumed_rank_arr(x) result(res)
			real(qp), dimension(..), intent(in) :: x

			res = .false.
			select rank(x)
			rank(1)
				if (size(x) >= 2) res = (x(2) == 2.0_qp)
			rank(2)
				if (size(x,1) >= 2 .and. size(x,2) >= 1) res = (x(2,1) == 2.0_qp)
			rank default
				res = .false.
			end select

		end function func_qp_assumed_rank_arr


		logical function func_qcmplx_qp_explicit_arr_1d(x) result(res)
			complex(qp), dimension(4), intent(inout) :: x

			res = .false.
			if (x(2) == (2.0_qp, -2.0_qp)) res = .true.
			x = x + (1.0_qp, 1.0_qp)

		end function func_qcmplx_qp_explicit_arr_1d


		logical function func_qcmplx_qp_assumed_shape_arr_1d(x) result(res)
			complex(qp), dimension(:), intent(inout) :: x

			res = .false.
			if (size(x) >= 2) res = (x(2) == (2.0_qp, -2.0_qp))
			x = x + (2.0_qp, 0.0_qp)

		end function func_qcmplx_qp_assumed_shape_arr_1d


		logical function func_qcmplx_qp_assumed_size_arr_1d(x, n) result(res)
			complex(qp), intent(inout) :: x(*)
			integer, intent(in) :: n

			res = .false.
			if (n >= 2) res = (x(2) == (2.0_qp, -2.0_qp))
			x(1:n) = x(1:n) + (3.0_qp, -3.0_qp)

		end function func_qcmplx_qp_assumed_size_arr_1d


		logical function func_qcmplx_qp_assumed_rank_arr(x) result(res)
			complex(qp), dimension(..), intent(in) :: x

			res = .false.
			select rank(x)
			rank(1)
				if (size(x) >= 2) res = (x(2) == (2.0_qp, -2.0_qp))
			rank(2)
				if (size(x,1) >= 2 .and. size(x,2) >= 1) res = (x(2,1) == (2.0_qp, -2.0_qp))
			rank default
				res = .false.
			end select

		end function func_qcmplx_qp_assumed_rank_arr


		real(qp) function func_test_quad_ret()
			
			func_test_quad_ret = 3.14_qp

		end function  func_test_quad_ret

		function func_qp_return_array() result(res)
			real(qp), dimension(4) :: res

			res = [1.0_qp, 2.0_qp, 3.0_qp, 4.0_qp]

		end function func_qp_return_array

		function func_qp_return_alloc_array(n) result(res)
			integer, intent(in) :: n
			real(qp), allocatable, dimension(:) :: res
			integer :: i

			allocate(res(n))
			do i = 1, n
				res(i) = real(10 * i, kind=qp)
			end do

		end function func_qp_return_alloc_array

		function func_qp_return_from_assumed_shape(x) result(res)
			real(qp), dimension(:), intent(in) :: x
			real(qp), dimension(size(x)) :: res

			res = x + 5.0_qp

		end function func_qp_return_from_assumed_shape

		function func_qp_return_from_assumed_size(x, n) result(res)
			real(qp), intent(in) :: x(*)
			integer, intent(in) :: n
			real(qp), dimension(n) :: res

			res = x(1:n) + 7.0_qp

		end function func_qp_return_from_assumed_size

		function func_qcmplx_qp_return_array() result(res)
			complex(qp), dimension(4) :: res

			res = [ &
				cmplx(1.0_qp, -1.0_qp, kind=qp), &
				cmplx(2.0_qp, -2.0_qp, kind=qp), &
				cmplx(3.0_qp, -3.0_qp, kind=qp), &
				cmplx(4.0_qp, -4.0_qp, kind=qp)  &
			]

		end function func_qcmplx_qp_return_array

		function func_qcmplx_qp_return_alloc_array(n) result(res)
			integer, intent(in) :: n
			complex(qp), allocatable, dimension(:) :: res
			integer :: i

			allocate(res(n))
			do i = 1, n
				res(i) = cmplx(real(10 * i, kind=qp), real(-10 * i, kind=qp), kind=qp)
			end do

		end function func_qcmplx_qp_return_alloc_array

		function func_qcmplx_qp_return_from_assumed_shape(x) result(res)
			complex(qp), dimension(:), intent(in) :: x
			complex(qp), dimension(size(x)) :: res

			res = x + cmplx(5.0_qp, -5.0_qp, kind=qp)

		end function func_qcmplx_qp_return_from_assumed_shape

		function func_qcmplx_qp_return_from_assumed_size(x, n) result(res)
			complex(qp), intent(in) :: x(*)
			integer, intent(in) :: n
			complex(qp), dimension(n) :: res

			res = x(1:n) + cmplx(7.0_qp, -7.0_qp, kind=qp)

		end function func_qcmplx_qp_return_from_assumed_size

#endif

end module quad
