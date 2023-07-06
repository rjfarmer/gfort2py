! SPDX-License-Identifier: GPL-2.0+

module dt

    use iso_fortran_env, only: output_unit, real128
    
    implicit none
    
    ! Parameters
    integer, parameter :: dp = selected_real_kind(p=15)
    integer, parameter :: qp = selected_real_kind(p=30)
    integer, parameter :: lp = selected_int_kind(16)
    
    
    TYPE s_struct_basic
        integer           :: a_int
        integer(lp)       :: a_int_lp
        real              :: a_real
        real(dp)          :: a_real_dp
        character(len=10) :: a_str    
        integer, dimension(5) :: b_int_exp_1d 
        real(dp), dimension(5) :: b_real_dp_exp_1d 
        integer, allocatable, dimension(:) :: c_int_alloc_1d
        integer, pointer, dimension(:) :: d_int_point_1d => null()
    END TYPE s_struct_basic
    
    TYPE s_simple
        integer           :: x,y
    END TYPE s_simple
    
    
    TYPE s_struct_nested
        integer           :: a_int
        TYPE(s_struct_basic) :: f_struct
    END TYPE s_struct_nested
    
    TYPE s_struct_nested_2
        integer           :: a_int
        TYPE(s_struct_nested) :: f_nested
    END TYPE s_struct_nested_2
    
    TYPE(s_simple) :: f_struct_simple
    
    TYPE s_recursive
        integer           :: a_int
        TYPE(s_recursive),pointer :: s_recur
    end TYPE s_recursive
    
    TYPE s_recursive_2
        integer           :: a_int
        TYPE(s_recursive_1),pointer :: s_recur
    end TYPE s_recursive_2  
    
    TYPE s_recursive_1
        integer           :: a_int
        TYPE(s_recursive_2),pointer :: s_recur
    end TYPE s_recursive_1
    
    TYPE s_alloc_array
        integer           :: a_int
        real              :: a_real
        real(dp)          :: a_real_dp
        integer, dimension(0:15) :: bb
        character(len=20) :: a_str   
        integer,dimension(:,:,:),allocatable :: arr
        integer           :: a_int2
        real(dp), allocatable, dimension(:,:) :: alloc_arr
    END TYPE s_alloc_array
    
    
    TYPE(s_struct_basic) :: f_struct
    TYPE(s_struct_basic),dimension(2) :: f_struct_exp_1d
    TYPE(s_struct_basic),dimension(2,2) :: f_struct_exp_2d
    TYPE(s_struct_basic),dimension(:),  allocatable :: f_struct_alloc_1d
    TYPE(s_struct_basic),dimension(:,:),allocatable :: f_struct_alloc_2d
    TYPE(s_struct_basic),dimension(:),  pointer :: f_struct_point_1d => null()
    TYPE(s_struct_basic),dimension(:,:),pointer :: f_struct_point_2d => null()
    TYPE(s_struct_basic),dimension(2),  target :: f_struct_target_1d
    TYPE(s_struct_basic),dimension(2,2),target :: f_struct_target_2d
    
    
    TYPE(s_struct_nested) :: g_struct
    
    TYPE(s_struct_basic),dimension(2) :: g_struct_exp_1d
    TYPE(s_struct_basic),dimension(2,2) :: g_struct_exp_2d
    TYPE(s_struct_basic),dimension(:),  allocatable :: g_struct_alloc_1d
    TYPE(s_struct_basic),dimension(:,:),allocatable :: g_struct_alloc_2d
    TYPE(s_struct_basic),dimension(:),  pointer :: g_struct_point_1d => null()
    TYPE(s_struct_basic),dimension(:,:),pointer :: g_struct_point_2d => null()
    TYPE(s_struct_basic),dimension(2),  target :: g_struct_target_1d
    TYPE(s_struct_basic),dimension(2,2),target :: g_struct_target_2d
    
    
    TYPE(s_struct_nested_2) :: h_struct
    TYPE(s_struct_nested_2),dimension(2) :: h_struct_exp_1d
    TYPE(s_struct_nested_2),dimension(2,2) :: h_struct_exp_2d
    TYPE(s_struct_nested_2),dimension(:),  allocatable :: h_struct_alloc_1d
    TYPE(s_struct_nested_2),dimension(:,:),allocatable :: h_struct_alloc_2d
    TYPE(s_struct_nested_2),dimension(:),  pointer :: h_struct_point_1d => null()
    TYPE(s_struct_nested_2),dimension(:,:),pointer :: h_struct_point_2d => null()
    TYPE(s_struct_nested_2),dimension(2),  target :: h_struct_target_1d
    TYPE(s_struct_nested_2),dimension(2,2),target :: h_struct_target_2d
    
    
    TYPE(s_recursive) :: r_recur
    TYPE(s_recursive_1) :: r_recur_1
    TYPE(s_recursive_2) :: r_recur_2
    
    
    integer, target, dimension(5) :: e_int_target_1d
    
    !GH: #32
    type :: point 
        integer, dimension(4) :: iq
    end type point

    
    contains


 
    subroutine sub_f_simple_in(x)
        type(s_simple), intent(in) :: x
        
        write(output_unit,'(2(I2,1X))') x%x,x%y
    end subroutine sub_f_simple_in
    
    subroutine sub_f_simple_out(x)
        type(s_simple), intent(out) :: x
        
        x%x=1
        x%y=10
    end subroutine sub_f_simple_out
    
    subroutine sub_f_simple_inout(zzz)
        type(s_simple), intent(inout) :: zzz
        
        write(output_unit,'(2(I2,1X))') zzz%x,zzz%y
        zzz%x=1
        zzz%y=10
    end subroutine sub_f_simple_inout
    
    subroutine sub_f_simple_inoutp(zzz)
        type(s_simple),pointer, intent(inout) :: zzz
        
        write(output_unit,'(2(I2,1X))') zzz%x,zzz%y
        zzz%x=1
        zzz%y=10
    end subroutine sub_f_simple_inoutp
    
    
    logical function func_check_nested_dt() result(res10)
    
        if(g_struct%a_int==10 .and. &
            g_struct%f_struct%a_int==8) then
            res10=.True.
        else
            write(*,*) g_struct%a_int, g_struct%f_struct%a_int
            res10=.false.
        end if
    
    end function func_check_nested_dt      
    
    
    subroutine sub_f_struct_simple()
    
        write(*,*) f_struct_simple%x,f_struct_simple%y
        f_struct_simple%x =22
        f_struct_simple%y =55
    
    end subroutine sub_f_struct_simple
    
    
    subroutine sub_dt_alloc_ar(x)
        type(s_alloc_array) :: x
        
        allocate(x%alloc_arr(1:10,1:10))
        x%alloc_arr=99.d0
    
    end subroutine sub_dt_alloc_ar
    
    
    logical function func_set_f_struct()
        f_struct%a_int=5
        f_struct%a_int_lp=6_lp         
        f_struct%a_real = 7.0
        f_struct%a_real_dp=8.0_dp
        f_struct%a_str='9999999999'
        f_struct%b_int_exp_1d=(/9,10,11,12,13/)
        
        if (allocated(f_struct%c_int_alloc_1d)) deallocate(f_struct%c_int_alloc_1d)
        allocate(f_struct%c_int_alloc_1d(1:10))
        f_struct%c_int_alloc_1d=(/1,2,3,4,5,6,7,8,9,10/)
        
        e_int_target_1d = (/9,10,11,12,13/)
        nullify(f_struct%d_int_point_1d)
        f_struct%d_int_point_1d => e_int_target_1d
        
        func_set_f_struct = .true.
    
    end function func_set_f_struct
    
    logical function func_check_f_struct()
    
        write(*,*)         f_struct%a_int,loc(f_struct%a_int)
        write(*,*)         f_struct%a_int_lp  ,loc(f_struct%a_int_lp)      
        write(*,*)         f_struct%a_real,loc(f_struct%a_real)
        write(*,*)         f_struct%a_real_dp,loc(f_struct%a_real_dp)
        write(*,*)         f_struct%b_int_exp_1d,loc(f_struct%b_int_exp_1d)
        
        if (allocated(f_struct%c_int_alloc_1d))then
            write(*,*)      f_struct%c_int_alloc_1d,loc(f_struct%c_int_alloc_1d)
        end if
        
        if(associated(f_struct%d_int_point_1d))then
            write(*,*)      f_struct%d_int_point_1d,loc(f_struct%d_int_point_1d)
            write(*,*)      e_int_target_1d,loc(e_int_target_1d)
        end if        
    
    end function func_check_f_struct
    
    
    subroutine sub_s_struct_inout(s)
        type(s_struct_basic), intent(inout) :: s
        
        s% a_int = 99
        s% a_int_lp = 99_lp
        s% a_real = 99.0
        s% a_real_dp = 99.0_dp
        s% a_str ='1234567890'
        s% b_int_exp_1d = (/1,2,3,4,5/)
        s% b_real_dp_exp_1d = (/1.0,2.0,3.0,4.0,5.0/)
        if (.not. allocated(s% c_int_alloc_1d)) allocate(s% c_int_alloc_1d(1:10))
        s% c_int_alloc_1d = 99
    
    end subroutine sub_s_struct_inout
    
    
    subroutine sub_struct_exp_1d(x)
        TYPE(s_struct_basic),dimension(2) :: x
        
        x(1)%a_int = 5
        x(2)%a_int = 9
        
        x(1)%b_int_exp_1d = 66
        x(2)%b_int_exp_1d = 77
        
    end subroutine sub_struct_exp_1d
    

    logical function check_g_struct_exp_2d()
        
        check_g_struct_exp_2d = .false.
        if(g_struct_exp_2d(1,1)%a_int == 1 .and. &
            g_struct_exp_2d(2,1)%a_int == 2 .and. &
            g_struct_exp_2d(1,2)%a_int == 3 .and. &
            g_struct_exp_2d(2,2)%a_int == 4 ) then
            check_g_struct_exp_2d = .true.
        else
            write(*,*) g_struct_exp_2d(1,1)%a_int, g_struct_exp_2d(2,1)%a_int,g_struct_exp_2d(1,2)%a_int,g_struct_exp_2d(2,2)%a_int 
        end if
    
    
    end function check_g_struct_exp_2d


    logical function check_recur()

        check_recur = .false.
        
        if(r_recur%a_int==9 .and. &
            r_recur%s_recur%a_int==9 .and. &
            r_recur%s_recur%s_recur%a_int == 9) then
            check_recur = .true.
        else
            write(*,*) r_recur%a_int
            if(associated(r_recur%s_recur)) write(*,*) r_recur%s_recur%a_int
            if(associated(r_recur%s_recur%s_recur)) write(*,*) r_recur%s_recur%s_recur%a_int
        end if


    end function check_recur


    function func_return_s_struct_nested_2() result(s)
        type(s_struct_nested_2)::s
        s%a_int = 123
        s%f_nested%a_int = 234
        s%f_nested%f_struct%a_int = 345
    end function func_return_s_struct_nested_2


    subroutine derived_structure(p)
        ! GH: #32
        type(point), intent(out) :: p
        p = point([10,20,30,40])
        write(*,'(4(I2,1X))') p
    end subroutine derived_structure


end module dt
