module alloc

use iso_fortran_env
implicit none

type abc
integer :: x
end type
contains

subroutine allocate_integer_INT8_1(x, bounds)
integer(INT8),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_integer_INT8_1(x)
integer(INT8),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT8_2(x, bounds)
integer(INT8),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_integer_INT8_2(x)
integer(INT8),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT8_3(x, bounds)
integer(INT8),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_integer_INT8_3(x)
integer(INT8),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT8_4(x, bounds)
integer(INT8),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_integer_INT8_4(x)
integer(INT8),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT8_5(x, bounds)
integer(INT8),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_integer_INT8_5(x)
integer(INT8),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT16_1(x, bounds)
integer(INT16),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_integer_INT16_1(x)
integer(INT16),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT16_2(x, bounds)
integer(INT16),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_integer_INT16_2(x)
integer(INT16),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT16_3(x, bounds)
integer(INT16),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_integer_INT16_3(x)
integer(INT16),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT16_4(x, bounds)
integer(INT16),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_integer_INT16_4(x)
integer(INT16),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT16_5(x, bounds)
integer(INT16),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_integer_INT16_5(x)
integer(INT16),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT32_1(x, bounds)
integer(INT32),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_integer_INT32_1(x)
integer(INT32),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT32_2(x, bounds)
integer(INT32),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_integer_INT32_2(x)
integer(INT32),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT32_3(x, bounds)
integer(INT32),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_integer_INT32_3(x)
integer(INT32),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT32_4(x, bounds)
integer(INT32),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_integer_INT32_4(x)
integer(INT32),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT32_5(x, bounds)
integer(INT32),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_integer_INT32_5(x)
integer(INT32),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT64_1(x, bounds)
integer(INT64),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_integer_INT64_1(x)
integer(INT64),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT64_2(x, bounds)
integer(INT64),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_integer_INT64_2(x)
integer(INT64),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT64_3(x, bounds)
integer(INT64),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_integer_INT64_3(x)
integer(INT64),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT64_4(x, bounds)
integer(INT64),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_integer_INT64_4(x)
integer(INT64),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_integer_INT64_5(x, bounds)
integer(INT64),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_integer_INT64_5(x)
integer(INT64),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL32_1(x, bounds)
real(REAL32),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_real_REAL32_1(x)
real(REAL32),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL32_2(x, bounds)
real(REAL32),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_real_REAL32_2(x)
real(REAL32),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL32_3(x, bounds)
real(REAL32),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_real_REAL32_3(x)
real(REAL32),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL32_4(x, bounds)
real(REAL32),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_real_REAL32_4(x)
real(REAL32),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL32_5(x, bounds)
real(REAL32),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_real_REAL32_5(x)
real(REAL32),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL64_1(x, bounds)
real(REAL64),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_real_REAL64_1(x)
real(REAL64),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL64_2(x, bounds)
real(REAL64),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_real_REAL64_2(x)
real(REAL64),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL64_3(x, bounds)
real(REAL64),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_real_REAL64_3(x)
real(REAL64),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL64_4(x, bounds)
real(REAL64),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_real_REAL64_4(x)
real(REAL64),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL64_5(x, bounds)
real(REAL64),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_real_REAL64_5(x)
real(REAL64),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL128_1(x, bounds)
real(REAL128),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_real_REAL128_1(x)
real(REAL128),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL128_2(x, bounds)
real(REAL128),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_real_REAL128_2(x)
real(REAL128),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL128_3(x, bounds)
real(REAL128),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_real_REAL128_3(x)
real(REAL128),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL128_4(x, bounds)
real(REAL128),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_real_REAL128_4(x)
real(REAL128),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_real_REAL128_5(x, bounds)
real(REAL128),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_real_REAL128_5(x)
real(REAL128),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL32_1(x, bounds)
complex(REAL32),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_complex_REAL32_1(x)
complex(REAL32),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL32_2(x, bounds)
complex(REAL32),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_complex_REAL32_2(x)
complex(REAL32),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL32_3(x, bounds)
complex(REAL32),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_complex_REAL32_3(x)
complex(REAL32),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL32_4(x, bounds)
complex(REAL32),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_complex_REAL32_4(x)
complex(REAL32),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL32_5(x, bounds)
complex(REAL32),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_complex_REAL32_5(x)
complex(REAL32),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL64_1(x, bounds)
complex(REAL64),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_complex_REAL64_1(x)
complex(REAL64),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL64_2(x, bounds)
complex(REAL64),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_complex_REAL64_2(x)
complex(REAL64),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL64_3(x, bounds)
complex(REAL64),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_complex_REAL64_3(x)
complex(REAL64),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL64_4(x, bounds)
complex(REAL64),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_complex_REAL64_4(x)
complex(REAL64),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL64_5(x, bounds)
complex(REAL64),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_complex_REAL64_5(x)
complex(REAL64),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL128_1(x, bounds)
complex(REAL128),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_complex_REAL128_1(x)
complex(REAL128),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL128_2(x, bounds)
complex(REAL128),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_complex_REAL128_2(x)
complex(REAL128),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL128_3(x, bounds)
complex(REAL128),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_complex_REAL128_3(x)
complex(REAL128),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL128_4(x, bounds)
complex(REAL128),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_complex_REAL128_4(x)
complex(REAL128),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_complex_REAL128_5(x, bounds)
complex(REAL128),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_complex_REAL128_5(x)
complex(REAL128),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_logical_4_1(x, bounds)
logical(4),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_logical_4_1(x)
logical(4),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_logical_4_2(x, bounds)
logical(4),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_logical_4_2(x)
logical(4),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_logical_4_3(x, bounds)
logical(4),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_logical_4_3(x)
logical(4),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_logical_4_4(x, bounds)
logical(4),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_logical_4_4(x)
logical(4),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_logical_4_5(x, bounds)
logical(4),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_logical_4_5(x)
logical(4),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_character_4_1(x, bounds, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_character_4_1(x, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_character_4_2(x, bounds, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_character_4_2(x, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_character_4_3(x, bounds, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_character_4_3(x, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_character_4_4(x, bounds, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_character_4_4(x, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_character_4_5(x, bounds, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_character_4_5(x, n)
integer,intent(in) :: n
character(len=n),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_abc_4_1(x, bounds)
type(abc),allocatable :: x(:)
integer,intent(in) :: bounds(1)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1)))
end subroutine

subroutine deallocate_abc_4_1(x)
type(abc),allocatable :: x(:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_abc_4_2(x, bounds)
type(abc),allocatable :: x(:,:)
integer,intent(in) :: bounds(2)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2)))
end subroutine

subroutine deallocate_abc_4_2(x)
type(abc),allocatable :: x(:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_abc_4_3(x, bounds)
type(abc),allocatable :: x(:,:,:)
integer,intent(in) :: bounds(3)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3)))
end subroutine

subroutine deallocate_abc_4_3(x)
type(abc),allocatable :: x(:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_abc_4_4(x, bounds)
type(abc),allocatable :: x(:,:,:,:)
integer,intent(in) :: bounds(4)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4)))
end subroutine

subroutine deallocate_abc_4_4(x)
type(abc),allocatable :: x(:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

subroutine allocate_abc_4_5(x, bounds)
type(abc),allocatable :: x(:,:,:,:,:)
integer,intent(in) :: bounds(5)
if(allocated(x)) deallocate(x)
allocate(x(1:bounds(1),1:bounds(2),1:bounds(3),1:bounds(4),1:bounds(5)))
end subroutine

subroutine deallocate_abc_4_5(x)
type(abc),allocatable :: x(:,:,:,:,:)
if(allocated(x)) deallocate(x)
end subroutine

end module alloc
