! SPDX-License-Identifier: GPL-2.0+

module proc_ptrs
    use iso_fortran_env, only: output_unit, real128
    
    implicit none
    
    ! Parameters
    integer, parameter :: dp = selected_real_kind(p=15)
    integer, parameter :: qp = selected_real_kind(p=30)
    integer, parameter :: lp = selected_int_kind(16)
    
          
    procedure(func_func_run), pointer:: p_func_func_run_ptr => NULL()
    procedure(func_func_run), pointer:: p_func_func_run_ptr2 => func_func_run
    
    procedure(func_func_run_dp), pointer:: p_func_func_run_dp_ptr => NULL()


    contains
    
    
    integer function func_func_arg(func)
        integer :: func
        func_func_arg = func(1)
    end function func_func_arg
    
    
    real(dp) function func_func_arg_dp(z,func)
        integer :: z
        real(dp) :: func
        func_func_arg_dp = func(z)
    end function func_func_arg_dp
    
 
    real(dp) function func_real(x)
        integer(dp) :: x
        func_real = 100d0 * x 
    end function func_real
    
    
    integer function func_func_run(x)
        integer :: x
        !write(*,*) "x is",x,LOC(x)
        func_func_run = 10*x
        
    end function func_func_run   

    real(dp) function func_func_run_dp(x)
        real(dp) :: x
        !write(*,*) "x is",x,LOC(x)
        func_func_run_dp = 10*x
        
    end function func_func_run_dp
    
    integer function func_func_run2(x)
        integer :: x
        
        func_func_run2 = 2*x
    
    end function func_func_run2   
    
    
    integer function func_proc_ptr(x)
        integer :: x
        
        func_proc_ptr = p_func_func_run_ptr(x)
    end function func_proc_ptr
    
    subroutine sub_proc_ptr2()      
        p_func_func_run_ptr => func_func_run2
    end subroutine sub_proc_ptr2
    
    subroutine sub_null_proc_ptr()
        p_func_func_run_ptr => null()
    end subroutine sub_null_proc_ptr
    

    subroutine sub_proc_ptr_setter(x)      
        procedure(func_func_run) :: x

        p_func_func_run_ptr => x
    end subroutine sub_proc_ptr_setter

    subroutine sub_proc_ptr_setter2()      
        procedure(func_func_run), pointer:: x => func_func_run

        p_func_func_run_ptr => x
    end subroutine sub_proc_ptr_setter2

    
    integer function proc_proc_func_arg(x)
         procedure(func_func_run), pointer :: x
         
         proc_proc_func_arg = x(9)
    
    end function proc_proc_func_arg


end module proc_ptrs
