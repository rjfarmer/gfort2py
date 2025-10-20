# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import collections
from typing import List, Any, NamedTuple, Type
from functools import cache

import gfModParser as gf

from .arguments import fArguments
from .subroutine import fSub
from .functions import fFunc
from .procedures import fProcedure


def factory(procedure) -> fProcedure:
    if procedure.is_subroutine:
        return fSub
    else:
        return fFunc
