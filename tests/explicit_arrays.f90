! SPDX-License-Identifier: GPL-2.0+

module explicit_arrays

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	integer,parameter,dimension(10) :: const_int_arr=(/1,2,3,4,5,6,7,8,9,0/)
	real,parameter,dimension(10) :: const_real_arr=(/1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0/)
	real(dp),parameter,dimension(10) :: const_real_dp_arr=(/1_dp,2_dp,3_dp,4_dp,5_dp,6_dp,7_dp,8_dp,9_dp,0_dp/)
	logical, parameter, dimension(5) :: const_logical_arr=(/.true.,.false.,.true.,.false.,.true./)

	! Arrays
	integer, dimension(5) :: b_int_exp_1d
	integer, dimension(5,5) :: b_int_exp_2d
	integer, dimension(5,5,5) :: b_int_exp_3d
	integer, dimension(5,5,5,5) :: b_int_exp_4d
	integer, dimension(5,5,5,5,5) :: b_int_exp_5d
	
	real, dimension(5) :: b_real_exp_1d
	real, dimension(5,5) :: b_real_exp_2d
	real, dimension(5,5,5) :: b_real_exp_3d
	real, dimension(5,5,5,5) :: b_real_exp_4d
	real, dimension(5,5,5,5,5) :: b_real_exp_5d
	
	real(dp), dimension(5) :: b_real_dp_exp_1d
	real(dp), dimension(5,5) :: b_real_dp_exp_2d
	real(dp), dimension(5,5,5) :: b_real_dp_exp_3d
	real(dp), dimension(5,5,5,5) :: b_real_dp_exp_4d
	real(dp), dimension(5,5,5,5,5) :: b_real_dp_exp_5d
	real(dp), dimension(2,3,4,5,6) :: b_real_exp_5d_2


	character(len=10), dimension(5) :: b_str_exp_1d
	character(len=10), dimension(5,5) :: b_str_exp_2d
	character(len=10), dimension(5,5,5) :: b_str_exp_3d
	character(len=10), dimension(5,5,5,5) :: b_str_exp_4d
	character(len=10), dimension(5,5,5,5,5) :: b_str_exp_5d     


	integer, dimension(5) :: b_int_exp_1d_set = (/1,2,3,4,5/)
	
	
	contains


	subroutine sub_array_n_int_1d(n,x)
		integer,intent(in) :: n
		integer,dimension(n), intent(in) :: x
		
		write(output_unit,'(5(I1,1X))') x
	end subroutine sub_array_n_int_1d
	
	subroutine sub_array_n_int_2d(n,m,x)
		integer,intent(in) :: n, m
		integer,dimension(n,m), intent(in) :: x
		
		write(output_unit,'(25(I1,1X))') x
	end subroutine sub_array_n_int_2d
	
	subroutine sub_exp_array_int_1d(x)
		integer,dimension(5), intent(in) :: x
		
		write(output_unit,'(5(I1,1X))') x
	end subroutine sub_exp_array_int_1d
	
	subroutine sub_exp_array_int_2d(x)
		integer,dimension(5,5), intent(in) :: x
		integer :: i,j,k
		write(output_unit,'(25(I2.2,1X))') x
	end subroutine sub_exp_array_int_2d
	
	subroutine sub_exp_array_int_3d(x)
		integer,dimension(5,5,5), intent(in) :: x
		write(output_unit,'(125(I3.3,1X))') x
	end subroutine sub_exp_array_int_3d
	
	subroutine sub_exp_array_real_1d(x)
		real,dimension(5), intent(in) :: x
		write(output_unit,'(5(F5.1,1X))') x
	end subroutine sub_exp_array_real_1d
	
	subroutine sub_exp_array_real_2d(x)
		real,dimension(5,5), intent(in) :: x
		write(output_unit,'(25(F5.1,1X))') x
	end subroutine sub_exp_array_real_2d  
	
	subroutine sub_exp_array_real_3d(x)
		real,dimension(5,5,5), intent(in) :: x
		write(output_unit,'(125(F5.1,1X))') x
	end subroutine sub_exp_array_real_3d
	
	subroutine sub_exp_array_real_dp_1d(x)
		real(dp),dimension(5), intent(in) :: x
		write(output_unit,'(5(F5.1,1X))') x
	end subroutine sub_exp_array_real_dp_1d
	
	subroutine sub_exp_array_real_dp_2d(x)
		real(dp),dimension(5,5), intent(in) :: x
		write(output_unit,'(25(F5.1,1X))') x
	end subroutine sub_exp_array_real_dp_2d  
	
	subroutine sub_exp_array_real_dp_3d(x)
		real(dp),dimension(5,5,5), intent(in) :: x
		write(output_unit,'(125(F5.1,1X))') x
	end subroutine sub_exp_array_real_dp_3d
	
	subroutine sub_exp_array_int_1d_multi(y,x,z)
		integer,dimension(5), intent(in) :: x
		integer,intent(in) :: y,z
		
		write(output_unit,'(I2,1X,5(I1,1X),I2,1X)') y,x,z
	end subroutine sub_exp_array_int_1d_multi

	subroutine sub_exp_inout(x)
	 integer,dimension(5),intent(inout) :: x
	 x=2*x
	end subroutine sub_exp_inout   
      
	logical function func_logical_multi(a,b,x,c,d) result(res2)
		real(dp),intent(in) :: a,b,c,d
		logical, dimension(5), intent(in) :: x
		res2 =.false.
		if(all(x.eqv..true.)) res2 = .true.
	end function func_logical_multi
	
	
	function func_mesh_exp(N) result(mesh)
		! Github issues #13
		integer, intent(in) :: N
		integer(dp) :: mesh(N+1)
		integer :: i
		
		do i=1,n+1
			mesh(i) = i
		end do
		
	end function func_mesh_exp


	subroutine func_mesh_exp2(x,N) 
		integer, intent(in) :: N
		integer :: x(N+1)
		integer :: i
		
		do i=1,n+1
			x(i) = i
		end do
		
	end subroutine func_mesh_exp2


	subroutine func_mesh_exp3(x,N) 
		integer, intent(in) :: N
		integer :: x((N*2)+1)
		integer :: i
		
		do i=1,(N*2)+1
			x(i) = i
		end do
		
	end subroutine func_mesh_exp3


	subroutine func_mesh_exp4(x,N) 
		integer, intent(in) :: N
		integer :: x((N+3)*2+1)
		integer :: i
		
		do i=1,(N+3)*2+1
			x(i) = i
		end do
		
	end subroutine func_mesh_exp4
	
	
	subroutine check_exp_2d_2m3(arr, success)
		! Github issues #19
		integer, dimension(2,3) :: arr
		logical :: success
		
		success=.false.
		
		if(arr(1,2)==1 .and. arr(2,1)==2 .and. arr(2,3) == 3 &
		.and. arr(1,1)==0 .and. arr(2,2)==0 .and. arr(1,3) ==0) success = .true.
		

	end subroutine check_exp_2d_2m3
	
	
	subroutine check_exp_2d_2m3_nt(arr, NT, success)
		! Github issues #19
		integer, intent(in) :: NT
		integer, dimension(3,NT) :: arr
		logical :: success
		integer :: i
		
		success=.false.
		
		if(arr(1,2)==1 .and. arr(2,1)==2 .and. arr(2,3) == 3 .and. arr(2,4) ==4 ) success = .true.
		
!		do i=1,3
!			write(*,*) arr(i,:)
!		end do
		
		arr(1,NT) = 5

	end subroutine check_exp_2d_2m3_nt	
	


	function func_return_1d_int_arr() result(x)
		integer,dimension(5) :: x

		x = (/1,2,3,4,5/)

	end function func_return_1d_int_arr

	function func_return_1d_int_arr_n(n) result(x)
		integer, intent(in) :: n
		integer,dimension(n) :: x
		integer :: i

		do i=1,n
			x(i) = i
		end do

	end function func_return_1d_int_arr_n


	function func_return_2d_int_arr() result(x)
		integer,dimension(3,2) :: x

		x = reshape((/1,2,3,4,5,6/), shape(x))

	end function func_return_2d_int_arr


	subroutine func_exp_array_in(n,x)
		integer, intent(in) :: n
		integer, dimension(2*n,2**n) :: x

		x = 5

	end subroutine func_exp_array_in


end module explicit_arrays
