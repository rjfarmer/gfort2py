! SPDX-License-Identifier: GPL-2.0+

module basic2
	implicit none
	integer, parameter :: lp2 = selected_int_kind(8)

	integer :: test2_x


end module basic2



module basic
	use iso_fortran_env, only: output_unit, real128
	use basic2

	implicit none
	
	     ! Parameters
		integer, parameter :: dp = selected_real_kind(p=15)
		integer, parameter :: qp = selected_real_kind(p=30)
		integer, parameter :: lp = selected_int_kind(16)
		
		integer, parameter      :: const_int=1
		integer, parameter      :: const_int_p1=const_int+1
		integer(lp), parameter  :: const_int_lp=1_lp

		logical,parameter :: const_logical_true=.true.
		logical,parameter :: const_logical_false=.false.

		
		real, parameter     :: const_real=1.0
		real(dp), parameter :: const_real_dp=1.0_dp
		real(dp), parameter :: const_real_pi_dp=3.14_dp

		integer, parameter :: const_neg_int = -1.0
		real(dp),parameter :: const_neg_real = -3.14_dp
			
		! Variables
		integer           :: a_int
		integer(lp)       :: a_int_lp
		real              :: a_real
		real(dp)          :: a_real_dp
		
		! Set variables
		integer           :: a_int_set=5
		integer(lp)       :: a_int_lp_set=6
		real              :: a_real_set=7.0
		real(dp)          :: a_real_dp_set=8.0_dp

		! Test Mixed Case variables
		integer :: a_int_MIXED
		integer :: const_int_MIXED = 1

		
	contains


		subroutine sub_no_args()
			write(output_unit,*) "1"
		end subroutine sub_no_args
		
		integer function func_int_no_args()
			func_int_no_args=2
			! write(output_unit,*) 2
		end function func_int_no_args
		
		real function func_real_no_args()
			func_real_no_args=3.0
			! write(output_unit,*) 3.0
		end function func_real_no_args
		
		real(dp) function func_real_dp_no_args()
			func_real_dp_no_args=4.0_dp
			! write(output_unit,*) 4.0_dp
		end function func_real_dp_no_args
		
		subroutine sub_int_in(x)
			integer, intent(in) ::x
			write(output_unit,*) 2*x
		end subroutine sub_int_in
		
		subroutine sub_int_out(x)
			integer, intent(out) :: x
			x=1
		end subroutine sub_int_out
		
		subroutine sub_int_inout(x)
			integer, intent(inout) :: x
			x=2*x
		end subroutine sub_int_inout    
		
		subroutine sub_real_inout(x)
			real, intent(inout) :: x
			x=2*x
		end subroutine sub_real_inout   
		
		subroutine sub_multi_inout(x,y,z)
			integer, intent(in) :: x
			integer, intent(inout) :: y
			integer, intent(out) :: z
			y=y*x
			z=y*x
		end subroutine sub_multi_inout  
		
		subroutine sub_int_no_intent(x)
			integer :: x
			x=2*x
		end subroutine sub_int_no_intent 
		
		integer function func_int_in(x)
			integer, intent(in) :: x
			func_int_in=2*x
		end function func_int_in
		
		integer function func_int_in_multi(x,y,z)
			integer, intent(in) ::x,y,z
			func_int_in_multi=x+y+z
		end function func_int_in_multi
		
		subroutine sub_alter_mod()
			a_int=99
			a_int_lp=99_lp
			a_real=99.0
			a_real_dp=99.0_dp
		end subroutine sub_alter_mod
      
		logical function func_check_mod()
			func_check_mod = .false.
		
			if( a_int==5 .and. &
				a_int_lp==5_lp .and. &
				a_real==5.0 .and. &
			    a_real_dp==5.0_dp) then
			    
			    func_check_mod = .true.
			end if

		end function func_check_mod
      
      
		integer function func_intent_out(y,x)
			integer, intent(in) :: y
			integer, intent(out) :: x
		
			func_intent_out = y
			x =y
		
		end function func_intent_out
		
		function func_result(y,x) result(z)
			integer, intent(in) :: y
			integer, intent(out) :: x
			integer :: z
		
			z = y*2
			x =y
		
		end function func_result
		
		
		function func_int_value(x) result(z)
			integer, intent(in),value :: x
			integer :: z
		
			z = x*2
		
		end function func_int_value

		subroutine sub_int_opt(x)
			integer, optional, intent(in) :: x
		
			if(present(x)) then
				write(*,*) 100
			else
				write(*,*) 200
			end if  
		end subroutine sub_int_opt
    

		subroutine sub_int_opt_val(x)
			integer, optional, intent(in),value :: x
		
			if(present(x)) then
				write(*,*) 100
			else
				write(*,*) 200
			end if  
		end subroutine sub_int_opt_val
      
		logical function func_return_res(x) result(res1)
			integer, intent(in) :: x
			res1 =.false.
			if(x==2) res1 = .true.
		end function func_return_res
		

		subroutine sub_int_p(zzz)
			integer,pointer, intent(inout) :: zzz
			
			write(output_unit,'(I2)') zzz
			
			zzz = 5
		end subroutine sub_int_p
		
		subroutine sub_real_p(zzz)
			real,pointer, intent(inout) :: zzz
			
			write(output_unit,'(F5.2)') zzz
			
			zzz = 5.0
		end subroutine sub_real_p
		
		subroutine sub_many_args(a,b,c,d,e,f,g,h,i,j,k,l)
			real :: a,b,c,d
			logical :: e,f,g
			character(len=*) :: h,i,j,k,l
			
			!write(*,*) a,b,c,d,e,f,g,h,i,j,k,l
		end subroutine sub_many_args
		
		
		subroutine sub_use_mod()
		
			test2_x = 1
		
		end subroutine sub_use_mod
   

		logical function func_test_bool(x)
			integer,intent(in) :: x

			if(x==0) then
				func_test_bool = .false.
			else
				func_test_bool = .true.
			end if

		end function func_test_bool


		integer function func_TEST_CASE()
			func_TEST_CASE = 1
		end function func_TEST_CASE


end module basic
