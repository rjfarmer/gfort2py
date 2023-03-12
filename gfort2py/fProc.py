# SPDX-License-Identifier: GPL-2.0+
import ctypes
import os
import select
import collections

from .fVar_t import fVar_t
from .fUnary import run_unary


_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None


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
    Result = collections.namedtuple("Result", ["res", "args"])

    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self.obj = self._allobjs[key]
        self._lib = lib

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

    @property
    def __doc__(self):
        return f"Procedure"

    def __call__(self, *args, **kwargs):

        self._set_return()

        func_args = self._convert_args(*args, **kwargs)

        with _captureStdOut() as cs:
            if func_args is not None:
                res = self._func(*func_args)
            else:
                res = self._func()

        return self._convert_result(res, func_args)

    def _set_return(self):
        if self.obj.is_subroutine():
            self._func.restype = None  # Subroutine
        else:
            fvar = fVar_t(self._allobjs[self.obj.return_arg()])

            if (
                fvar.obj.is_char()
            ):  # Return a character is done as a character + len at start of arg list
                self._func.restype = None
            else:
                self._func.restype = fvar.ctype()

    def _convert_args(self, *args, **kwargs):

        res_start = []
        res = []
        res_end = []

        if self.obj.is_function():
            fvar = fVar_t(self._allobjs[self.obj.return_arg()])
            if fvar.obj.is_char():
                l = fvar.len()
                res_start.append(fvar.from_param(" " * l.value))
                res_start.append(l)

        count = 0
        input_args = []
        # Build list of inputs
        for fval in self.obj.args():
            var = fVar_t(self._allobjs[fval.ref])

            try:
                x = kwargs[var.obj.name]
            except KeyError:
                if count <= len(args):
                    x = args[count]
                    count = count + 1
                else:
                    raise TypeError("Not enough arguments passed")

            if x is None and not var.obj.is_optional() and not var.obj.is_dummy():
                raise ValueError(f"Got None for {var.name}")

            input_args.append((x, var))

        # Resolve unary operations

        # Convert to ctypes
        for x, var in input_args:
            if x is not None or var.obj.is_dummy():
                if var.obj.is_optional() and x is None:
                    res.append(None)
                else:
                    z = var.from_param(x)

                    if var.obj.is_value():
                        res.append(z)
                    elif var.obj.is_pointer():
                        if var.obj.not_a_pointer():
                            # print(self.name, var.obj.name, z)
                            res.append(ctypes.pointer(z))
                        else:
                            res.append(ctypes.pointer(ctypes.pointer(z)))
                    else:
                        # print(z)
                        res.append(ctypes.pointer(z))

                    if var.obj.is_defered_len():
                        res_end.append(var.len(x))
                    if var.obj.is_optional_value():
                        ct = ctypes.c_byte
                        res_end.append(ct(1))

            else:
                res.append(None)
                if var.obj.is_defered_len():
                    res_end.append(None)
                if var.obj.is_optional_value():
                    ct = ctypes.c_byte
                    res_end.append(ct(0))

        return res_start + res + res_end

    def _convert_result(self, result, args):
        res = {}

        if self.obj.is_function():
            fvar = fVar_t(self._allobjs[self.obj.return_arg()])
            if fvar.obj.is_char():
                result = args[0]
                _ = args.pop(0)
                _ = args.pop(0)  # Twice to pop first and second value

        if len(self.obj.args()):
            for ptr, fval in zip(args, self.obj.args()):
                res[self._allobjs[fval.ref].head.name] = fVar_t(
                    self._allobjs[fval.ref]
                ).from_ctype(ptr)

        if self.obj.is_function():
            result = fVar_t(self._allobjs[self.obj.return_arg()]).from_ctype(result)

        return self.Result(result, res)

    def __repr__(self):
        return self.__doc__

    @property
    def __doc__(self):

        if self.obj.is_subroutine():
            ftype = f"subroutine {self.name}"
        else:
            fv = fVar_t(self._allobjs[self.obj.return_arg()]).typekind
            ftype = f"{fv} function {self.name}"

        args = []
        for fval in self.obj.args():
            args.append(fVar_t(self._allobjs[fval.ref]).__doc__)

        args = ", ".join(args)
        return f"{ftype} ({args})"
