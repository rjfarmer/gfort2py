# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import select
import collections
import functools
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


class _captureStdOut:
    def read_pipe(self, pipe_out):
        def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)

        out = b""
        while more_data():
            out += os.read(pipe_out, 1024)
        return out.decode()

    def __enter__(self):
        if _TEST_FLAG:
            self.pipe_out, self.pipe_in = os.pipe()
            self.stdout = os.dup(1)
            os.dup2(self.pipe_in, 1)

    def __exit__(self, *args, **kwargs):
        if _TEST_FLAG:
            os.dup2(self.stdout, 1)
            print(self.read_pipe(self.pipe_out))
            os.close(self.pipe_in)
            os.close(self.pipe_out)
            os.close(self.stdout)


class fProc:
    Result = collections.namedtuple("Result", ["result", "args"])

    def __init__(self, lib, obj, allobjs, **kwargs):
        self._allobjs = allobjs
        self.obj = obj
        self._lib = lib
        self._return_value = None

        self._func = getattr(lib, self.mangled_name)

    @property
    def mangled_name(self):
        return self.obj.mangled_name

    def in_dll(self, lib):
        return self._func

    @property
    def module(self):
        return self.obj.head.module

    @property
    def name(self):
        return self.obj.name

    def __call__(self, *args, **kwargs):
        func_args = self._convert_args(*args, **kwargs)

        with _captureStdOut() as cs:
            if func_args is not None:
                res = self._func(*func_args)
            else:
                res = self._func()

        return self._convert_result(res, func_args)

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
        # Build list of inputs
        for fval in self.obj.args():
            var = fVar(self._allobjs[fval.ref], allobjs=self._allobjs)

            try:
                x = kwargs[var.name]
            except KeyError:
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

            arguments.append(variable(x, var, fval.ref))

        return arguments

    def args_convert(self, input_args):
        args = []
        args_end = []
        # Convert to ctypes
        for var in input_args:
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
            ftype = f"{self.return_var.__doc__()} function {self.name}"

        args = []
        for fval in self.obj.args():
            args.append(fVar(fval.ref, allobjs=self._allobjs).__doc__)

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
