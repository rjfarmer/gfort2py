
module unicode
    implicit none

    integer,parameter :: CK = selected_char_kind('ISO_10646')

    character(kind=ck,len=100),parameter :: uni_param = ck_'😀😎😩'

    character(kind=ck,len=100) :: uni_set = ck_'😀😎😩'





end module unicode