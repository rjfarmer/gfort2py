# SPDX-License-Identifier: GPL-2.0+
from .subroutine import fSub
from .functions import fFunc
from .procedures import fProcedure


def factory(procedure) -> fProcedure:
    if procedure.is_subroutine:
        return fSub
    else:
        return fFunc
