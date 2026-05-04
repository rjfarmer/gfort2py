! SPDX-License-Identifier: GPL-2.0+

module smod

	use iso_fortran_env, only: output_unit, real128
	
	implicit none
	
	! Parameters
	integer, parameter :: dp = selected_real_kind(p=15)
	integer, parameter :: qp = selected_real_kind(p=30)
	integer, parameter :: lp = selected_int_kind(16)
	
	
	interface 
		module function func_smod(x,y) result(z)
			real(dp), intent(in) :: x,y
			real(dp) :: z
		end function func_smod
		
		module function func_smod2(x,y) result(z)
			real(dp), intent(in) :: x,y
			real(dp) :: z
		end function func_smod2
		
	end interface
	
	
	
	contains



end module smod


submodule (smod) smod_s


	contains
	
		module function func_smod(x,y) result(z)
			real(dp), intent(in) :: x,y
			real(dp) :: z
			z = x+y
		end function func_smod
		
		
		module procedure func_smod2
			z = x+y
		end procedure func_smod2

end submodule smod_s
