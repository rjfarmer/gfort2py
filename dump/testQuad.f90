program testQuad

    implicit none

    integer, parameter :: qp = selected_real_kind(p=30)


    write(*,'(B128)') 0_qp
    write(*,'(B128)') 1_qp
    write(*,'(B128)') 2.1_qp
    write(*,'(Z128)') 2.1_qp
    write(*,'(B128)') 3.14569877412356447896914555666873990_qp
    write(*,'(Z128)') 3.14569877412356447896914555666873990_qp
    write(*,*)
    write(*,'(B128)') -0_qp
    write(*,'(B128)') -1_qp
    write(*,'(B128)') -2_qp
    write(*,'(B128)') -3.14569877412356447896914555666873990_qp
    write(*,*)

    write(*,'(B128)') -0.1_qp
    write(*,'(B128)') -0.2_qp
    write(*,'(B128)') -3.14569877412356447896914555666873990_qp

end program testQuad