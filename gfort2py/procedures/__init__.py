# SPDX-License-Identifier: GPL-2.0+
from .functions import fFunc
from .procedures import fProcedure
from .subroutine import fSub


def factory(procedure) -> fProcedure:
    if procedure.is_subroutine:
        return fSub
    else:
        return fFunc
