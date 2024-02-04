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
		
		! Set variables
		real(qp)          :: a_real_qp_set=9.0_qp

		complex(qp), parameter  :: const_cmplx_qp=(1.0_qp,1.0_qp)
		complex(qp)       :: a_cmplx_qp
		complex(qp)       :: a_cmplx_qp_set

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


		real(qp) function func_test_quad_ret()
			
			func_test_quad_ret = 3.14_qp

		end function  func_test_quad_ret

#endif

end module quad
