module leak_regression
  implicit none
contains
  ! Plain assumed-shape array dummy argument, used to exercise
  ! ftype_assumed_shape.ctype (the array-descriptor path).
  subroutine bump_assumed_shape_2d(a)
    real(8), intent(inout) :: a(:,:)
    a = a + 1.0d0
  end subroutine bump_assumed_shape_2d
end module leak_regression
