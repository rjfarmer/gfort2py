! SPDX-License-Identifier: GPL-2.0+

module oo

    use iso_fortran_env, only: output_unit, real128
    
    implicit none
    
    ! Parameters
    integer, parameter :: dp = selected_real_kind(p=15)
    integer, parameter :: qp = selected_real_kind(p=30)
    integer, parameter :: lp = selected_int_kind(16)
    
    
    TYPE s_proc
        integer :: a_int
        
        contains
        
        procedure, nopass :: proc_no_pass => func_dt_no_pass
        procedure, pass(this) :: proc_pass => sub_dt_pass
    
    end type s_proc
    
    
    type, extends(s_proc) :: s_proc_extend
        real(dp) :: a_real_dp
    
    end type s_proc_extend
    
    
    TYPE(s_proc) :: p_proc
    
    type(s_proc_extend) :: p_proc_extend
    
    contains
    
    integer function func_dt_no_pass(x)
        integer :: x
        
        func_dt_no_pass = 5*x
    
    end function func_dt_no_pass
    
    
    subroutine sub_dt_pass(this,x)
        class(s_proc), intent(inout) :: this
        integer :: x
        
        this%a_int = 5*x
        
    end subroutine sub_dt_pass
      


end module oo
