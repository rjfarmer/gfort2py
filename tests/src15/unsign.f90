module unsign
    implicit none

    unsigned :: a_u
    unsigned,parameter :: a_u_param = 32u
    unsigned,dimension(5) :: a_u_array_1d
    unsigned,dimension(5),parameter :: a_u_array_param_1d = (/1u,2u,3u,4u,5u/)

    contains

    subroutine set_unsigned_scalar(x, value)
        unsigned, intent(out) :: x
        unsigned, intent(in) :: value

        x = value
    end subroutine set_unsigned_scalar

    function add_unsigned(lhs, rhs) result(total)
        unsigned, intent(in) :: lhs
        unsigned, intent(in) :: rhs
        unsigned :: total

        total = lhs + rhs
    end function add_unsigned

    function max_unsigned(lhs, rhs) result(res)
        unsigned, intent(in) :: lhs
        unsigned, intent(in) :: rhs
        unsigned :: res

        if (lhs > rhs) then
            res = lhs
        else
            res = rhs
        end if
    end function max_unsigned

    subroutine copy_unsigned_array(src, dst)
        unsigned, intent(in) :: src(:)
        unsigned, intent(out) :: dst(:)

        dst = src
    end subroutine copy_unsigned_array

    subroutine scale_unsigned_array(arr, factor)
        unsigned, intent(inout) :: arr(:)
        unsigned, intent(in) :: factor

        arr = arr * factor
    end subroutine scale_unsigned_array

    function sum_unsigned_array(arr) result(total)
        unsigned, intent(in) :: arr(:)
        unsigned :: total
        integer :: i

        total = 0u
        do i = 1, size(arr)
            total = total + arr(i)
        end do
    end function sum_unsigned_array

    function add_unsigned_arrays(lhs, rhs) result(res)
        unsigned, intent(in) :: lhs(:)
        unsigned, intent(in) :: rhs(:)
        unsigned :: res(size(lhs))

        res = lhs + rhs
    end function add_unsigned_arrays

    function shift_unsigned_array(arr, shift) result(res)
        unsigned, intent(in) :: arr(:)
        unsigned, intent(in) :: shift
        unsigned :: res(size(arr))

        res = arr + shift
    end function shift_unsigned_array

end module unsign