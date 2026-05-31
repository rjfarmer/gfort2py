! SPDX-License-Identifier: GPL-2.0+
! This file is auto generated do not edit by hand
module vars_modules
    implicit none

    integer, parameter :: r4 = selected_real_kind(4)
    integer, parameter :: r8 = selected_real_kind(8)
    integer, parameter :: r16 = selected_real_kind(32)

    integer, parameter :: i1 = selected_int_kind(1)
    integer, parameter :: i2 = selected_int_kind(2)
    integer, parameter :: i4 = selected_int_kind(4)
    integer, parameter :: i8 = selected_int_kind(8)

    integer, parameter :: c1 = selected_char_kind ("ascii")
    integer, parameter :: c4  = selected_char_kind ('ISO_10646')
 

    integer(i1) :: int_i1_0 = -1_i1 

    integer(i1) :: int_i1_1 = 0_i1 

    integer(i1) :: int_i1_2 = 1_i1 

    integer(i2) :: int_i2_0 = -1_i2 

    integer(i2) :: int_i2_1 = 0_i2 

    integer(i2) :: int_i2_2 = 1_i2 

    integer(i4) :: int_i4_0 = -1_i4 

    integer(i4) :: int_i4_1 = 0_i4 

    integer(i4) :: int_i4_2 = 1_i4 

    integer(i8) :: int_i8_0 = -1_i8 

    integer(i8) :: int_i8_1 = 0_i8 

    integer(i8) :: int_i8_2 = 1_i8 

    real(r4) :: real_r4_0 = -3.140000104904175_r4 

    real(r4) :: real_r4_1 = 0_r4 

    real(r4) :: real_r4_2 = 3.140000104904175_r4 

    real(r8) :: real_r8_0 = -3.140000104904175_r8 

    real(r8) :: real_r8_1 = 0_r8 

    real(r8) :: real_r8_2 = 3.140000104904175_r8 

    
contains

 

    function check_int_i1_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i1_0/=-2_i1) return
    x = .true.
    end function check_int_i1_0 

    function check_int_i1_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i1_1/=0_i1) return
    x = .true.
    end function check_int_i1_1 

    function check_int_i1_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i1_2/=2_i1) return
    x = .true.
    end function check_int_i1_2 

    function check_int_i2_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i2_0/=-2_i2) return
    x = .true.
    end function check_int_i2_0 

    function check_int_i2_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i2_1/=0_i2) return
    x = .true.
    end function check_int_i2_1 

    function check_int_i2_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i2_2/=2_i2) return
    x = .true.
    end function check_int_i2_2 

    function check_int_i4_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i4_0/=-2_i4) return
    x = .true.
    end function check_int_i4_0 

    function check_int_i4_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i4_1/=0_i4) return
    x = .true.
    end function check_int_i4_1 

    function check_int_i4_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i4_2/=2_i4) return
    x = .true.
    end function check_int_i4_2 

    function check_int_i8_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i8_0/=-2_i8) return
    x = .true.
    end function check_int_i8_0 

    function check_int_i8_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i8_1/=0_i8) return
    x = .true.
    end function check_int_i8_1 

    function check_int_i8_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(int_i8_2/=2_i8) return
    x = .true.
    end function check_int_i8_2 

    function check_real_r4_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r4_0/=-6.28000020980835_r4) return
    x = .true.
    end function check_real_r4_0 

    function check_real_r4_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r4_1/=0_r4) return
    x = .true.
    end function check_real_r4_1 

    function check_real_r4_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r4_2/=6.28000020980835_r4) return
    x = .true.
    end function check_real_r4_2 

    function check_real_r8_0() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r8_0/=-6.28000020980835_r8) return
    x = .true.
    end function check_real_r8_0 

    function check_real_r8_1() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r8_1/=0_r8) return
    x = .true.
    end function check_real_r8_1 

    function check_real_r8_2() result(x) 
    implicit none
    logical :: x
    x = .false.
    if(real_r8_2/=6.28000020980835_r8) return
    x = .true.
    end function check_real_r8_2 

end module vars_modules
 

