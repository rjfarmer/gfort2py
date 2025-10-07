# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import collections
from typing import List, Any
from functools import cache

import gfModParser as gf
from .types import factory

Result = collections.namedtuple("Result", ["result", "args"])


class fProc:

    def __init__(self, lib, obj, module, **kwargs):
        self._module = module
        self.obj = obj
        self._lib = lib

        self._proc = getattr(self._lib, self.obj.mangled_name)

    def __call__(self, *args, **kwargs) -> Result:
        self._args = args
        self._kwargs = kwargs

        # Set return type
        self._set_return()

        # Set arguments
        self._set_arguments()

        # Call procedure
        self._call_procedure()

        # Convert return type
        ret = self._convert_return()

        # Convert arguments
        args = self._convert_arguments()

        return Result(ret, args)

    def _set_return(self):

        # Assume return is None, subroutines return as none
        self._proc = None

        # Procedures need values accessing via their number not name

        if self.obj.is_function:
            # If we are returning a character or array that gets added to the arguments
            # not the return value.
            if self.return_var.ftype == "character" or self._module[key].is_array():
                return

            # Quad's cant currently be returned
            if self.return_var.ftype == "real" and self.return_var.kind == 16:
                raise TypeError(
                    "Can not return a quad from a procedure, it must be an argument or module variable"
                )

            self._proc.restype = arg

    def _set_arguments(self):
        """Sets the procedures argument types"""
        self._proc.argtypes = (
            self._pre_set_args() + self._set_args() + self._post_set_args()
        )

    def _pre_set_args(self) -> List[Any]:
        """Arguments that get added to the start of the argument list"""
        return []

    def _set_args(self) -> List[Any]:
        """Arguments that go in the middle"""
        return []

    def _post_set_args(self) -> List[Any]:
        """Arguments that get added to the end of the argument list"""
        return []

    def _call_procedure(self):
        pass

    def _convert_return(self):
        pass

    def _convert_arguments(self):
        pass

    @property
    def __doc__(self):
        if self.obj.is_subroutine():
            ftype = f"subroutine {self.obj.name}"
        else:
            ftype = f"{self.return_var.__doc__} function {self.obj.name}"

        # args = []
        # for fval in self.obj.args():
        #     args.append(fVar(self._allobjs[fval.ref], allobjs=self._allobjs).__doc__)

        # args = ", ".join(args)
        # return f"{ftype} ({args})"
        return ftype

    @property
    @cache
    def return_var(self):
        key = self._module.properties.symbol_reference
        return factory(self._module[key])
