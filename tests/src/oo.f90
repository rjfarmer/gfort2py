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
        procedure, pass(this) :: proc_pass2 => sub_dt_pass2
    
        end type s_proc
    
    
    type, extends(s_proc) :: s_proc_extend
        real(dp) :: a_real_dp
    
    end type s_proc_extend
    
    
    TYPE(s_proc) :: p_proc
    TYPE(s_proc), dimension(3) :: p_proc_arr
    
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

    subroutine sub_dt_pass2(x,this)
        class(s_proc), intent(inout) :: this
        integer :: x
        
        this%a_int = 5*x
        
    end subroutine sub_dt_pass2

    subroutine sub_set_p_proc(x)
        integer, intent(in) :: x

        p_proc%a_int = x
    end subroutine sub_set_p_proc

    integer function func_get_p_proc()
        func_get_p_proc = p_proc%a_int
    end function func_get_p_proc

    subroutine sub_set_p_proc_extend(ai, ar)
        integer, intent(in) :: ai
        real(dp), intent(in) :: ar

        p_proc_extend%a_int = ai
        p_proc_extend%a_real_dp = ar
    end subroutine sub_set_p_proc_extend

    subroutine sub_get_p_proc_extend(ai, ar)
        integer, intent(out) :: ai
        real(dp), intent(out) :: ar

        ai = p_proc_extend%a_int
        ar = p_proc_extend%a_real_dp
    end subroutine sub_get_p_proc_extend

    subroutine sub_set_p_proc_arr(vals)
        integer, intent(in), dimension(3) :: vals

        p_proc_arr(1)%a_int = vals(1)
        p_proc_arr(2)%a_int = vals(2)
        p_proc_arr(3)%a_int = vals(3)
    end subroutine sub_set_p_proc_arr

    subroutine sub_get_p_proc_arr(vals)
        integer, intent(out), dimension(3) :: vals

        vals(1) = p_proc_arr(1)%a_int
        vals(2) = p_proc_arr(2)%a_int
        vals(3) = p_proc_arr(3)%a_int
    end subroutine sub_get_p_proc_arr

    function func_return_obj(x) result(res)
        integer, intent(in) :: x
        type(s_proc) :: res

        res%a_int = x
    end function func_return_obj

    function func_return_obj_array() result(res)
        type(s_proc), dimension(2) :: res

        res(1)%a_int = 11
        res(2)%a_int = 22
    end function func_return_obj_array

    function func_return_obj_array_alloc(n) result(res)
        integer, intent(in) :: n
        type(s_proc), allocatable, dimension(:) :: res
        integer :: i

        allocate(res(n))
        do i = 1, n
            res(i)%a_int = 100 + i
        end do
    end function func_return_obj_array_alloc

    subroutine sub_class_set_get(this, x, y)
        class(s_proc), intent(inout) :: this
        integer, intent(in) :: x
        integer, intent(out) :: y

        this%a_int = x
        y = this%a_int
    end subroutine sub_class_set_get

    function func_return_class(x, as_extended) result(res)
        integer, intent(in) :: x
        logical, intent(in) :: as_extended
        class(s_proc), allocatable :: res

        if (as_extended) then
            allocate(s_proc_extend :: res)
            res%a_int = x
            select type(res)
            type is (s_proc_extend)
                res%a_real_dp = real(x, dp) * 10.0_dp
            end select
        else
            allocate(s_proc :: res)
            res%a_int = x
        end if
    end function func_return_class

    subroutine sub_fill_class_array(arr, base)
        class(s_proc), intent(inout), dimension(:) :: arr
        integer, intent(in) :: base
        integer :: i

        do i = 1, size(arr)
            arr(i)%a_int = base + i
        end do
    end subroutine sub_fill_class_array

    logical function func_check_class_array(arr, base) result(ok)
        class(s_proc), intent(in), dimension(:) :: arr
        integer, intent(in) :: base
        integer :: i

        ok = .true.
        do i = 1, size(arr)
            if (arr(i)%a_int /= (base + i)) then
                ok = .false.
                return
            end if
        end do
    end function func_check_class_array

    subroutine sub_make_class_array(arr, n, as_extended)
        class(s_proc), allocatable, intent(out), dimension(:) :: arr
        integer, intent(in) :: n
        logical, intent(in) :: as_extended
        integer :: i

        if (as_extended) then
            allocate(s_proc_extend :: arr(n))
            do i = 1, n
                arr(i)%a_int = 200 + i
            end do
            select type(arr)
            type is (s_proc_extend)
                do i = 1, n
                    arr(i)%a_real_dp = real(1000 + i, dp)
                end do
            end select
        else
            allocate(s_proc :: arr(n))
            do i = 1, n
                arr(i)%a_int = 300 + i
            end do
        end if
    end subroutine sub_make_class_array
      


end module oo
