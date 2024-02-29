# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import collections
from dataclasses import dataclass

from .fVar import fVar
from .fVar_t import fVar_t
from .utils import resolve_other_args

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


@dataclass
class variable:
    value: "typing.Any"
    fvar: "typing.Any"
    symbol_ref: int


class fProc:
    Result = collections.namedtuple("Result", ["result", "args"])

    def __init__(self, lib, obj, allobjs, **kwargs):
        self._allobjs = allobjs
        self.obj = obj
        self._lib = lib
        self._return_value = None

        self._func = getattr(self._lib, self.mangled_name)

    @property
    def mangled_name(self):
        return self.obj.mangled_name

    def in_dll(self, lib):
        return self._func

    def from_param(self, *args):
        return self._func

    @property
    def module(self):
        return self.obj.head.module

    @property
    def value(self):
        return self._func

    @property
    def name(self):
        return self.obj.name

    def __call__(self, *args, **kwargs):
        # Reset return value
        self._return_value = None

        self._func_args = self._convert_args(*args, **kwargs)

        # print(self._func_args)

        if self._func_args is not None:
            res = self._func(*self._func_args)
        else:
            res = self._func()

        return self._convert_result(res, self._func_args)

    def _set_return(self, other_args):
        if self.obj.is_subroutine():
            self._func.restype = None  # Subroutine
        else:
            self.return_var.obj = resolve_other_args(
                self.return_var.obj, other_args, self._allobjs, self._lib, fProc
            )

            if self.return_var.obj.is_returned_as_arg():
                self._func.restype = None
            else:
                if self.return_var.obj.kind == 16:
                    raise TypeError(
                        "Can not return a quad from a procedure, it must be an argument or module variable"
                    )

                self._func.restype = self.return_var.ctype()

    def args_start(self):
        res = []
        if self.obj.is_function():
            if self.return_var.obj.is_char():
                l = self.return_var.len()
                res.append(self.return_var.from_param(" " * l))
                res.append(self.return_var.ctype_len())
            elif self.return_var.obj.is_always_explicit() and self.obj.is_array():
                empty_array = self.return_var._make_empty()
                res.append(ctypes.pointer(self.return_var.from_param(empty_array)))

        return res

    def args_check(self, *args, **kwargs):
        count = 0
        arguments = []

        if len(args) + len(kwargs) < len(self.obj.args()):
            raise TypeError("Not enough arguments passed")
        elif len(args) + len(kwargs) > len(self.obj.args()):
            raise TypeError("too many arguments passed")

        # Build list of inputs
        for fval in self.obj.args():
            if self._allobjs[fval.ref].is_procedure():
                var = None
                name = self._allobjs[fval.ref].name
            else:
                var = fVar(self._allobjs[fval.ref], allobjs=self._allobjs)
                name = var.name

            try:
                x = kwargs[name]
            except (KeyError, AttributeError):
                if count <= len(args):
                    x = args[count]
                    count = count + 1
                else:
                    raise TypeError("Not enough arguments passed")

            if x is None and not var.obj.is_optional() and not var.obj.is_dummy():
                raise ValueError(f"Got None for {var.name}")

            if isinstance(x, fVar_t):
                var = x
                x = var.value

            if isinstance(x, fProc):
                var = x

            arguments.append(variable(x, var, fval.ref))

        return arguments

    def args_convert(self, input_args):
        args = []
        args_end = []
        # Convert to ctypes
        for var in input_args:
            if var.fvar.obj.is_procedure():
                a = var.value.value
                e = None
                # print(a.value)
            else:
                _, a, e = var.fvar.to_proc(var.value, input_args)
            args.append(a)
            if e is not None:
                args_end.append(e)

        return args, args_end

    def _convert_args(self, *args, **kwargs):
        self.input_args = self.args_check(*args, **kwargs)

        for var in self.input_args:
            var.fvar.obj = resolve_other_args(
                var.fvar.obj, self.input_args, self._allobjs, self._lib, fProc
            )

        # Set this now after args_check as we need to be able to
        # resolve runtime arguments
        self._set_return(self.input_args)
        args_start = self.args_start()

        args_mid, args_end = self.args_convert(self.input_args)

        return args_start + args_mid + args_end

    def _convert_result(self, result, args):
        res = {}

        if self.obj.is_function():
            if self.return_var.obj.is_returned_as_arg():
                if self.return_var.obj.is_char():
                    result = args[0]
                    _ = args.pop(0)
                    _ = args.pop(0)  # Twice to pop first and second value
                elif self.return_var.obj.is_always_explicit() and self.obj.is_array():
                    result = self.return_var.value
                    _ = args.pop(0)

        if len(self.obj.args()):
            for var in self.input_args:
                if var.fvar.obj.is_procedure():
                    x = var.value
                else:
                    if var.fvar.unpack:
                        try:
                            x = ptr_unpack(var.fvar.value)
                        except AttributeError:  # unset optional arguments
                            x = None
                    else:
                        x = var.fvar.cvalue

                name = self._allobjs[var.symbol_ref].name
                if hasattr(x, "_type_"):
                    res[name] = var.fvar.from_ctype(x)
                else:
                    res[name] = x

        if self.obj.is_function():
            result = self.return_var.from_ctype(result)

        return self.Result(result, res)

    def __repr__(self):
        return self.__doc__

    @property
    def __doc__(self):
        if self.obj.is_subroutine():
            ftype = f"subroutine {self.name}"
        else:
            ftype = f"{self.return_var.__doc__} function {self.name}"

        args = []
        for fval in self.obj.args():
            args.append(fVar(self._allobjs[fval.ref], allobjs=self._allobjs).__doc__)

        args = ", ".join(args)
        return f"{ftype} ({args})"

    @property
    def return_var(self):
        if self._return_value is None:
            self._return_value = fVar(
                self._allobjs[self.obj.return_arg()], allobjs=self._allobjs
            )
        return self._return_value


def ptr_unpack(ptr):
    x = ptr
    if hasattr(ptr, "contents"):
        if hasattr(ptr.contents, "contents"):
            x = ptr.contents.contents
        else:
            x = ptr.contents
    return x
