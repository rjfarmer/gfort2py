# SPDX-License-Identifier: GPL-2.0+
import collections
import ctypes
import os
from functools import cache
from typing import Any, List, NamedTuple, Type

import gfModParser as gf

from ..types import f_type
from ..types import factory as type_factory
from .arguments import fArguments
from .procedures import fProcedure


class fFunc(fProcedure):

    def _set_return(self):
        # Procedures need values accessing via their number not name

        ftype = self.return_type.type.lower()
        kind = self.return_type.kind
        # If we are returning a character or array that gets added to the arguments
        # not the return value.
        if ftype == "character" or self.return_type.is_array:
            self._proc.restype = None
            return

        # Quad's cant currently be returned
        if ftype == "real" and kind == 16:
            raise TypeError(
                "Can not return a quad from a procedure, it must be an argument, module variable, or returned as an array"
            )

        self._proc.restype = self.return_var.ctype

    @property
    @cache
    def return_type(self) -> gf.Symbol:
        key = self.definition.properties.symbol_reference
        return self._module[key]

    @property
    @cache
    def return_var(self) -> f_type:
        cls = type_factory(self.return_type)
        c = cls.__new__(cls)
        c._symbol = self.return_type
        c._module_obj = self._module
        type(c).__init__(c)  # type: ignore[misc]
        return c

    @property
    def __doc__(self):
        ftype = f"{str(self.return_var)} function {self.definition.name}"

        return f"{ftype} ({self._doc_args()})"

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        if self.return_type.type.lower() == "character" or self.return_type.is_array:
            self._result = None
            return

        self.return_var._ctype = value
        self._result = self.return_var.value
