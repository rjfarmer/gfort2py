module unicode
    implicit none

    integer, parameter :: CK = selected_char_kind('ISO_10646')

    character(kind=ck,len=100), parameter :: uni_param = ck_'😀😎😩'

    character(kind=ck,len=100) :: uni_set = ck_'😀😎😩'
    character(kind=ck,len=100), dimension(3) :: uni_arr = [ck_'😀😀😀', ck_'😎😎😎', ck_'😩😩😩']
    character(kind=ck,len=100), dimension(:), allocatable :: uni_alloc_arr

contains

    subroutine sub_set_uni_set(x)
        character(kind=ck,len=*), intent(in) :: x
        uni_set = x
    end subroutine sub_set_uni_set

    subroutine sub_uni_echo(x, y)
        character(kind=ck,len=*), intent(in) :: x
        character(kind=ck,len=100), intent(out) :: y
        y = x
    end subroutine sub_uni_echo

    function func_uni_ret() result(y)
        character(kind=ck,len=100) :: y
        y = ck_'🌍🚀'
    end function func_uni_ret

    subroutine sub_uni_arr_inout(x)
        character(kind=ck,len=*), dimension(3), intent(inout) :: x

        x(1) = ck_'😀😎😩'
        x(2) = ck_'漢字'
        x(3) = ck_'aΩβ'
    end subroutine sub_uni_arr_inout

    logical function func_uni_assumed_shape_ok(x) result(res)
        character(kind=ck,len=*), dimension(:), intent(in) :: x

        res = .false.

        if (size(x) /= 3) return
        if (x(1) /= ck_'🚀🚀🚀') return
        if (x(2) /= ck_'🌍🌍🌍') return
        if (x(3) /= ck_'✨✨✨') return

        res = .true.
    end function func_uni_assumed_shape_ok

    subroutine alloc_uni_alloc_arr()
        if (allocated(uni_alloc_arr)) deallocate(uni_alloc_arr)

        allocate(uni_alloc_arr(3))
        uni_alloc_arr(1) = ck_'😀😎😩'
        uni_alloc_arr(2) = ck_'漢字Ω'
        uni_alloc_arr(3) = ck_'aΩβ'
    end subroutine alloc_uni_alloc_arr

    logical function func_uni_assumed_rank_ok(x) result(res)
        character(kind=ck,len=*), dimension(..), intent(in) :: x

        res = .false.

        select rank (x)
        rank (1)
            if (size(x) /= 3) return
            if (x(1) /= ck_'🚀🚀🚀') return
            if (x(2) /= ck_'🌍🌍🌍') return
            if (x(3) /= ck_'✨✨✨') return
            res = .true.
        rank (2)
            if (size(x,1) /= 2 .or. size(x,2) /= 2) return
            if (x(1,1) /= ck_'🚀🚀🚀') return
            if (x(2,1) /= ck_'🌍🌍🌍') return
            if (x(1,2) /= ck_'✨✨✨') return
            if (x(2,2) /= ck_'漢字Ω') return
            res = .true.
        rank default
            res = .false.
        end select
    end function func_uni_assumed_rank_ok


    function func_return_alloc_unicode() result(x)
        character(kind=ck,len=:), allocatable :: x
        x = ck_'🚀🚀🚀'
    end function func_return_alloc_unicode

end module unicode