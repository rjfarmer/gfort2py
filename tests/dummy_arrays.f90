! SPDX-License-Identifier: GPL-2.0+

module dummy_arrays

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	integer, allocatable, dimension(:) :: c_int_alloc_1d
	integer, allocatable, dimension(:,:) :: c_int_alloc_2d
	integer, allocatable, dimension(:,:,:) :: c_int_alloc_3d
	integer, allocatable, dimension(:,:,:,:) :: c_int_alloc_4d
	integer, allocatable, dimension(:,:,:,:,:) :: c_int_alloc_5d
	
	real, allocatable, dimension(:) :: c_real_alloc_1d
	real, allocatable, dimension(:,:) :: c_real_alloc_2d
	real, allocatable, dimension(:,:,:) :: c_real_alloc_3d
	real, allocatable, dimension(:,:,:,:) :: c_real_alloc_4d
	real, allocatable, dimension(:,:,:,:,:) :: c_real_alloc_5d
	
	real(dp), allocatable, dimension(:) :: c_real_dp_alloc_1d
	real(dp), allocatable, dimension(:,:) :: c_real_dp_alloc_2d
	real(dp), allocatable, dimension(:,:,:) :: c_real_dp_alloc_3d
	real(dp), allocatable, dimension(:,:,:,:) :: c_real_dp_alloc_4d
	real(dp), allocatable, dimension(:,:,:,:,:) :: c_real_dp_alloc_5d
	
	
	integer, pointer, dimension(:) :: d_int_point_1d => null()
	integer, pointer, dimension(:,:) :: d_int_point_2d => null()
	integer, pointer, dimension(:,:,:) :: d_int_point_3d => null()
	integer, pointer, dimension(:,:,:,:) :: d_int_point_4d => null()
	integer, pointer, dimension(:,:,:,:,:) :: d_int_point_5d => null()
	
	real, pointer, dimension(:) :: d_real_point_1d => null()
	real, pointer, dimension(:,:) :: d_real_point_2d => null()
	real, pointer, dimension(:,:,:) :: d_real_point_3d => null()
	real, pointer, dimension(:,:,:,:) :: d_real_point_4d => null()
	real, pointer, dimension(:,:,:,:,:) :: d_real_point_5d => null()
	
	real(dp), pointer, dimension(:) :: d_real_dp_point_1d => null()
	real(dp), pointer, dimension(:,:) :: d_real_dp_point_2d => null()
	real(dp), pointer, dimension(:,:,:) :: d_real_dp_point_3d => null()
	real(dp), pointer, dimension(:,:,:,:) :: d_real_dp_point_4d => null()
	real(dp), pointer, dimension(:,:,:,:,:) :: d_real_dp_point_5d => null()
	
	integer, target, dimension(5) :: e_int_target_1d
	integer, target, dimension(5,5) :: e_int_target_2d
	integer, target, dimension(5,5,5) :: e_int_target_3d
	integer, target, dimension(5,5,5,5) :: e_int_target_4d
	integer, target, dimension(5,5,5,5,5) :: e_int_target_5d
	
	real, target, dimension(5) :: e_real_target_1d
	real, target, dimension(5,5) :: e_real_target_2d
	real, target, dimension(5,5,5) :: e_real_target_3d
	real, target, dimension(5,5,5,5) :: e_real_target_4d
	real, target, dimension(5,5,5,5,5) :: e_real_target_5d
	
	real(dp), target, dimension(5) :: e_real_dp_target_1d
	real(dp), target, dimension(5,5) :: e_real_dp_target_2d
	real(dp), target, dimension(5,5,5) :: e_real_dp_target_3d
	real(dp), target, dimension(5,5,5,5) :: e_real_dp_target_4d
	real(dp), target, dimension(5,5,5,5,5) :: e_real_dp_target_5d
	
	
	character(len=10), allocatable, dimension(:) :: c_str_alloc_1d
	character(len=10), allocatable, dimension(:,:) :: c_str_alloc_2d
	character(len=10), allocatable, dimension(:,:,:) :: c_str_alloc_3d
	character(len=10), allocatable, dimension(:,:,:,:) :: c_str_alloc_4d
	character(len=10), allocatable, dimension(:,:,:,:,:) :: c_str_alloc_5d     

	
	contains
	
	
	subroutine sub_alloc_int_1d_arrs()
	
		if(.not. allocated(c_int_alloc_1d)) allocate(c_int_alloc_1d(5))
		if(.not. allocated(c_int_alloc_2d)) allocate(c_int_alloc_2d(5,5))
		if(.not. allocated(c_int_alloc_3d)) allocate(c_int_alloc_3d(5,5,5))
		if(.not. allocated(c_int_alloc_4d)) allocate(c_int_alloc_4d(5,5,5,5))
		if(.not. allocated(c_int_alloc_5d)) allocate(c_int_alloc_5d(5,5,5,5,5))
		
		c_int_alloc_1d=1
		c_int_alloc_2d=1
		c_int_alloc_3d=1
		c_int_alloc_4d=1  
		c_int_alloc_5d=1
	
	end subroutine sub_alloc_int_1d_arrs

	subroutine print_c_int_alloc_1d()
		write(*,*) c_int_alloc_1d
	end subroutine print_c_int_alloc_1d
	
	
	subroutine sub_alloc_int_1d_cleanup()
	
		if(allocated(c_int_alloc_1d)) deallocate(c_int_alloc_1d)
		if(allocated(c_int_alloc_2d)) deallocate(c_int_alloc_2d)
		if(allocated(c_int_alloc_3d)) deallocate(c_int_alloc_3d)
		if(allocated(c_int_alloc_4d)) deallocate(c_int_alloc_4d)
		if(allocated(c_int_alloc_5d)) deallocate(c_int_alloc_5d)
	
	end subroutine sub_alloc_int_1d_cleanup   
	
	
	subroutine sub_alter_int_1d()
	
		write(*,*) c_int_alloc_1d
		c_int_alloc_1d = 5
		write(*,*) c_int_alloc_1d
	
	end subroutine sub_alter_int_1d  
	
	
	subroutine sub_alloc_real_1d_arrs()
	
		if(.not. allocated(c_real_alloc_1d)) allocate(c_real_alloc_1d(5))
		if(.not. allocated(c_real_alloc_2d)) allocate(c_real_alloc_2d(5,5))
		if(.not. allocated(c_real_alloc_3d)) allocate(c_real_alloc_3d(5,5,5))
		if(.not. allocated(c_real_alloc_4d)) allocate(c_real_alloc_4d(5,5,5,5))
		if(.not. allocated(c_real_alloc_5d)) allocate(c_real_alloc_5d(5,5,5,5,5))
		
		c_real_alloc_1d=1
		c_real_alloc_2d=1
		c_real_alloc_3d=1
		c_real_alloc_4d=1  
		c_real_alloc_5d=1
	
	end subroutine sub_alloc_real_1d_arrs
	
	
	subroutine sub_alloc_real_1d_cleanup()
	
		if(allocated(c_real_alloc_1d)) deallocate(c_real_alloc_1d)
		if(allocated(c_real_alloc_2d)) deallocate(c_real_alloc_2d)
		if(allocated(c_real_alloc_3d)) deallocate(c_real_alloc_3d)
		if(allocated(c_real_alloc_4d)) deallocate(c_real_alloc_4d)
		if(allocated(c_real_alloc_5d)) deallocate(c_real_alloc_5d)
	
	end subroutine sub_alloc_real_1d_cleanup  
	
	subroutine sub_alloc_real_dp_1d_arrs()
	
		if(.not. allocated(c_real_dp_alloc_1d)) allocate(c_real_dp_alloc_1d(5))
		if(.not. allocated(c_real_dp_alloc_2d)) allocate(c_real_dp_alloc_2d(5,5))
		if(.not. allocated(c_real_dp_alloc_3d)) allocate(c_real_dp_alloc_3d(5,5,5))
		if(.not. allocated(c_real_dp_alloc_4d)) allocate(c_real_dp_alloc_4d(5,5,5,5))
		if(.not. allocated(c_real_dp_alloc_5d)) allocate(c_real_dp_alloc_5d(5,5,5,5,5))
		
		c_real_dp_alloc_1d=1
		c_real_dp_alloc_2d=1
		c_real_dp_alloc_3d=1
		c_real_dp_alloc_4d=1  
		c_real_dp_alloc_5d=1
	
	end subroutine sub_alloc_real_dp_1d_arrs
	
	
	subroutine sub_alloc_real_dp_1d_cleanup()
	
		if(allocated(c_real_dp_alloc_1d)) deallocate(c_real_dp_alloc_1d)
		if(allocated(c_real_dp_alloc_2d)) deallocate(c_real_dp_alloc_2d)
		if(allocated(c_real_dp_alloc_3d)) deallocate(c_real_dp_alloc_3d)
		if(allocated(c_real_dp_alloc_4d)) deallocate(c_real_dp_alloc_4d)
		if(allocated(c_real_dp_alloc_5d)) deallocate(c_real_dp_alloc_5d)
	
	end subroutine sub_alloc_real_dp_1d_cleanup  


	logical function func_alltrue_arr_1d(x) result(res2)
		logical, dimension(:), intent(in) :: x
		res2 =.false.
		if(all(x.eqv..true.)) res2 = .true.
		!         write(*,*) "1",x,"*",res2
		end function func_alltrue_arr_1d
	
	logical function func_allfalse_arr_1d(x) result(res2)
		logical, dimension(:), intent(in) :: x
		res2 =.false.
		if(all(x.eqv..false.)) res2 = .true.
		!         write(*,*) "2",x,"*",res2
	end function func_allfalse_arr_1d
	
	logical function func_allfalse_arr_1d_inout(x) result(res2)
		logical, dimension(:), intent(inout) :: x
		res2 =.false.
		if(all(x.eqv..false.)) res2 = .true.
		x=.True.
	end function func_allfalse_arr_1d_inout
	
	
	logical function func_assumed_shape_arr_1d(x) result(res2)
		integer, dimension(:), intent(inout) :: x
		res2 =.false.
		if(x(1)==2) res2 = .true.
		x=9
	end function func_assumed_shape_arr_1d
	
	logical function func_assumed_shape_arr_2d(x) result(res3)
		integer, dimension(:,:), intent(inout) :: x
		res3 =.false.
		if(x(2,1)==2) res3 = .true.
	end function func_assumed_shape_arr_2d
	
	logical function func_assumed_shape_arr_3d(x) result(res4)
		integer, dimension(:,:,:), intent(inout) :: x
		res4 =.false.
		if(x(3,2,1)==2) res4 = .true.
	end function func_assumed_shape_arr_3d
	
	logical function func_assumed_shape_arr_4d(x) result(res5)
		integer, dimension(:,:,:,:), intent(inout) :: x
		res5 =.false.
		if(x(4,3,2,1)==2) res5 = .true.
	end function func_assumed_shape_arr_4d
	
	logical function func_assumed_shape_arr_5d(x) result(res6)
		integer, dimension(:,:,:,:,:), intent(inout) :: x
		res6 =.false.
		if(x(5,4,3,2,1)==2) res6 = .true.
	end function func_assumed_shape_arr_5d
	
	
	logical function func_assumed_size_arr_1d(x) result(res7)
		integer, intent(inout) :: x(*)
		res7 =.false.
		if(x(2)==2) res7 = .true.
	end function func_assumed_size_arr_1d
	
	
	logical function func_assumed_size_arr_real_1d(x) result(res8)
		real, intent(inout) :: x(*)
		res8 =.false.
		if(x(2)==2) res8 = .true.
	end function func_assumed_size_arr_real_1d
	
	logical function func_assumed_size_arr_real_dp_1d(x) result(res9)
		real(dp), intent(inout) :: x(*)
		res9 =.false.
		if(x(2)==2) res9 = .true.
	end function func_assumed_size_arr_real_dp_1d 
	
	subroutine sub_alloc_arr_1d(x)
		integer, dimension(:),allocatable, intent(inout) :: x
		
		allocate(x(1:10))
		x=10
		
	end subroutine sub_alloc_arr_1d
	
	
	
	subroutine sub_arr_assumed_rank_int_1d(zzz)
		integer,dimension(:),pointer, intent(inout) :: zzz
		
		write(*,*) zzz(1:5)
		zzz(1:5) = 100
	end subroutine sub_arr_assumed_rank_int_1d
	
	subroutine sub_arr_assumed_rank_real_1d(zzz)
		real,dimension(:),pointer, intent(inout) :: zzz
		
		write(*,*) zzz(1:5)
		zzz(1:5) = 100.0
	end subroutine sub_arr_assumed_rank_real_1d
	
	subroutine sub_arr_assumed_rank_dp_1d(zzz)
		real(dp),dimension(:),pointer, intent(inout) :: zzz
		
		write(*,*) zzz(1:5),lbound(zzz),ubound(zzz),size(zzz)
		zzz(1:4) = 100.0_dp
		zzz(5) = 100_dp
	end subroutine sub_arr_assumed_rank_dp_1d



	subroutine sub_check_alloc_int_2d(x)
		integer, allocatable,dimension(:,:), intent(inout) :: x
		integer :: i,j
		
		if(.not.allocated(x)) allocate(x(3,4))
	
		do i=1,3
			do j=1,4
				x(i,j) = j + (i-1)*4
			 end do
		end do
	
	end subroutine sub_check_alloc_int_2d


	subroutine sub_check_alloc_int_3d(x)
		integer, allocatable,dimension(:,:,:), intent(inout) :: x
		integer :: i,j, k
		
		if(.not.allocated(x)) allocate(x(3,4,5))
	
		do i=1,3
			do j=1,4
				do k=1,5
					x(i,j,k) = k + (j-1)*4 + (i-1)*4 
				end do
			 end do
		end do
	
	end subroutine sub_check_alloc_int_3d


	function func_return_alloc_int_1d() result(v)

		integer,allocatable,dimension(:) :: v

		allocate(v(5))

		v = 1

	end function func_return_alloc_int_1d


	subroutine func2
		integer,allocatable,dimension(:) :: v

		v = func_return_alloc_int_1d()

	end subroutine func2


    !GH:39
    subroutine multi_array_pass(y, xp, yp)
        use iso_fortran_env, only: dp => real64
  
        real(dp), intent(inout) :: y
        real(dp), intent(inout) :: xp(:)
        real(dp), intent(inout) :: yp(:)
  
        y = -1000.0_dp
        xp = [13.0_dp, -2.0_dp]
        yp = [1.0_dp, -42.014_dp]
     end subroutine multi_array_pass



end module dummy_arrays
