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


end module comp
