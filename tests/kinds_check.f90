program kinds_check
    implicit none

    integer :: i,k


    do i=1,100
        k = selected_int_kind(i)
        if(k>0) write(*,*) 'Int ',i,k
    end do

    do i=1,100
        k = selected_real_kind(i)
        if(k>0) write(*,*) 'Real ',i,k
    end do


    write(*,*) 'Char ASCII',selected_char_kind('ASCII')
    write(*,*) 'Char DEFAULT',selected_char_kind('DEFAULT')
    write(*,*) 'Char ISO_10646',selected_char_kind('ISO_10646')



end program kinds_check

! Int            1           1
! Int            2           1
! Int            3           2
! Int            4           2
! Int            5           4
! Int            6           4
! Int            7           4
! Int            8           4
! Int            9           4
! Int           10           8
! Int           11           8
! Int           12           8
! Int           13           8
! Int           14           8
! Int           15           8
! Int           16           8
! Int           17           8
! Int           18           8
! Int           19          16
! Int           20          16
! Int           21          16
! Int           22          16
! Int           23          16
! Int           24          16
! Int           25          16
! Int           26          16
! Int           27          16
! Int           28          16
! Int           29          16
! Int           30          16
! Int           31          16
! Int           32          16
! Int           33          16
! Int           34          16
! Int           35          16
! Int           36          16
! Int           37          16
! Int           38          16
! Real            1           4
! Real            2           4
! Real            3           4
! Real            4           4
! Real            5           4
! Real            6           4
! Real            7           8
! Real            8           8
! Real            9           8
! Real           10           8
! Real           11           8
! Real           12           8
! Real           13           8
! Real           14           8
! Real           15           8
! Real           16          10
! Real           17          10
! Real           18          10
! Real           19          16
! Real           20          16
! Real           21          16
! Real           22          16
! Real           23          16
! Real           24          16
! Real           25          16
! Real           26          16
! Real           27          16
! Real           28          16
! Real           29          16
! Real           30          16
! Real           31          16
! Real           32          16
! Real           33          16
! Char ASCII           1
! Char DEFAULT           1
! Char ISO_10646           4