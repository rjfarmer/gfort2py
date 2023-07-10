! SPDX-License-Identifier: GPL-2.0+
! This file is auto generated do not edit by hand
module params_modules
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
 

    integer(i1),parameter :: int_i1_0 = -1_i1 

    integer(i1),parameter :: int_i1_1 = 0_i1 

    integer(i1),parameter :: int_i1_2 = 1_i1 

    integer(i2),parameter :: int_i2_0 = -1_i2 

    integer(i2),parameter :: int_i2_1 = 0_i2 

    integer(i2),parameter :: int_i2_2 = 1_i2 

    integer(i4),parameter :: int_i4_0 = -1_i4 

    integer(i4),parameter :: int_i4_1 = 0_i4 

    integer(i4),parameter :: int_i4_2 = 1_i4 

    integer(i8),parameter :: int_i8_0 = -1_i8 

    integer(i8),parameter :: int_i8_1 = 0_i8 

    integer(i8),parameter :: int_i8_2 = 1_i8 

    real(r4),parameter :: real_r4_0 = -3.140000104904175_r4 

    real(r4),parameter :: real_r4_1 = 0_r4 

    real(r4),parameter :: real_r4_2 = 3.140000104904175_r4 

    real(r8),parameter :: real_r8_0 = -3.140000104904175_r8 

    real(r8),parameter :: real_r8_1 = 0_r8 

    real(r8),parameter :: real_r8_2 = 3.140000104904175_r8 

    logical,parameter :: logicals_0 = .false. 

    logical,parameter :: logicals_1 = .true. 

    integer(i1),parameter,dimension(5) :: int_i1_1d = reshape( (/ -10, -1, 0, 1, 10/), shape(int_i1_1d)) 

    integer(i2),parameter,dimension(5) :: int_i2_1d = reshape( (/ -10, -1, 0, 1, 10/), shape(int_i2_1d)) 

    integer(i4),parameter,dimension(5) :: int_i4_1d = reshape( (/ -10, -1, 0, 1, 10/), shape(int_i4_1d)) 

    integer(i8),parameter,dimension(5) :: int_i8_1d = reshape( (/ -10, -1, 0, 1, 10/), shape(int_i8_1d)) 

    real(r4),parameter,dimension(3) :: real_r4_1d = reshape( (/ -3.140000104904175_r4, 0.0_r4, 3.140000104904175_r4/), shape(real_r4_1d)) 

    real(r8),parameter,dimension(3) :: real_r8_1d = reshape( (/ -3.140000104904175_r8, 0.0_r8, 3.140000104904175_r8/), shape(real_r8_1d)) 

    logical,parameter,dimension(4) :: logicals_0_1d = reshape( (/ .true., .false., .true., .false./), shape(logicals_0_1d)) 

    integer(i1),parameter,dimension(2,3) :: int_i1_2d = reshape( (/ -10, -1, 0, 1, 10, 50/), shape(int_i1_2d)) 

    integer(i2),parameter,dimension(2,3) :: int_i2_2d = reshape( (/ -10, -1, 0, 1, 10, 50/), shape(int_i2_2d)) 

    integer(i4),parameter,dimension(2,3) :: int_i4_2d = reshape( (/ -10, -1, 0, 1, 10, 50/), shape(int_i4_2d)) 

    integer(i8),parameter,dimension(2,3) :: int_i8_2d = reshape( (/ -10, -1, 0, 1, 10, 50/), shape(int_i8_2d)) 

    real(r4),parameter,dimension(2,3) :: real_r4_2d = reshape( (/ -6.28000020980835_r4, -3.140000104904175_r4, 0.0_r4, 1.1111_r4, 3.140000104904175_r4, 6.28000020980835_r4/), shape(real_r4_2d)) 

    real(r8),parameter,dimension(2,3) :: real_r8_2d = reshape( (/ -6.28000020980835_r8, -3.140000104904175_r8, 0.0_r8, 1.1111_r8, 3.140000104904175_r8, 6.28000020980835_r8/), shape(real_r8_2d)) 

    logical,parameter,dimension(2,3) :: logicals_0_2d = reshape( (/ .true., .false., .true., .false., .true., .false./), shape(logicals_0_2d)) 

    complex(r4),parameter :: complex_r4_0 = (-3.140000104904175_r4,-3.140000104904175_r4) 

    complex(r4),parameter :: complex_r4_1 = (0_r4,0_r4) 

    complex(r4),parameter :: complex_r4_2 = (3.140000104904175_r4,3.140000104904175_r4) 

    complex(r8),parameter :: complex_r8_0 = (-3.140000104904175_r8,-3.140000104904175_r8) 

    complex(r8),parameter :: complex_r8_1 = (0_r8,0_r8) 

    complex(r8),parameter :: complex_r8_2 = (3.140000104904175_r8,3.140000104904175_r8) 

    complex(r4),parameter,dimension(3) :: complex_r4_1d = reshape( (/ (-3.140000104904175,-3.140000104904175), (0,0), (3.140000104904175,3.140000104904175)/), shape(complex_r4_1d)) 

    complex(r8),parameter,dimension(3) :: complex_r8_1d = reshape( (/ (-3.140000104904175,-3.140000104904175), (0,0), (3.140000104904175,3.140000104904175)/), shape(complex_r8_1d)) 

    complex(r4),parameter,dimension(2,3) :: complex_r4_2d = reshape( (/ (-6.28000020980835,-6.28000020980835), (-3.140000104904175,-3.140000104904175), (0,0),(0,-1), (3.140000104904175,3.140000104904175),(-6.28000020980835,-6.28000020980835)/), shape(complex_r4_2d)) 

    complex(r8),parameter,dimension(2,3) :: complex_r8_2d = reshape( (/ (-6.28000020980835,-6.28000020980835), (-3.140000104904175,-3.140000104904175), (0,0),(0,-1), (3.140000104904175,3.140000104904175),(-6.28000020980835,-6.28000020980835)/), shape(complex_r8_2d)) 

end module params_modules
 

