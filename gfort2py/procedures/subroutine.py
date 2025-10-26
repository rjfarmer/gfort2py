# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import collections
from typing import List, Any, NamedTuple, Type
from functools import cache

import gfModParser as gf

from ..types import factory as type_factory

from .arguments import fArguments
from .procedures import fProcedure


class fSub(fProcedure):

    def _set_return(self):
        self._proc.argtypes = None

    @property
    def result(self):
        return None

    @result.setter
    def result(self, value):
        pass

    @property
    def __doc__(self):
        ftype = f"subroutine {self.definition.name}"

        return f"{ftype} ({self._doc_args()})"
