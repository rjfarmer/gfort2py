# SPDX-License-Identifier: GPL-2.0+
from __future__ import print_function

import ctypes
import os
import select
import collections

from .strings import fStr, fStrLen
from .errors import *
from .selector import _selectVar

_TEST_FLAG = os.environ.get("_GFORT2PY_TEST_FLAG") is not None

_allFuncs = {}

Result = collections.namedtuple('Result', 'result args')


class captureStdOut():
    def read_pipe(self, pipe_out):
        def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)
        out = b''
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


class fFunc(object):
    def __init__(self, obj):
        self.__dict__.update(obj)
        self._func = None
        self._extra_pre = []
        self._args = []
        self._args_in = []
        self._return = None
        self._isset = False

        self._DEBUG = False

    def in_dll(self, lib):
        self._sub = self.proc['sub']
        self.lib = lib

        self._args = self._init_args()
        self._init_return()
        self._func = getattr(lib, self.mangled_name)

        if not len(self._extra_pre):
            self._func.restype = self._return.ctype

        return self

    def __call__(self, *args):
        self._isSet(self.lib)

        if len(args) != len(self.arg):
            raise TypeError(str(self.name) +
                            " takes " +
                            str(len(self.arg)) +
                            " arguments got " +
                            str(len(args)))

        retstr = None
        retstrlen = -1
        args_in = []
        needs_extra = []
        extra_post = []

        start = 0
        end = 0
        # Handle strings at start of list (from function return types):
        if len(self._extra_pre):
            retstr = self._extra_pre[0].ctype()
            retstrlen = self._extra_pre[1].ctype(0)
            args_in = [retstr, retstrlen]
            start = 2
            needs_extra = [False, False]

        for value, ctype in zip(args, self._args[start:]):
            if isinstance(value, str) or isinstance(value, bytes):
                needs_extra.append(True)
            else:
                needs_extra.append(False)
            args_in.append(ctype.from_param(value))

        # Now handle adding string lengths to end of argument list
        for a in self._args[len(
                self.arg):]:  # self.arg is the list of normal arguments
            for v in args:
                if isinstance(v, str) or isinstance(v, bytes):
                    extra_post.append(a.from_param(v))

        end = start + len(self.arg)

        self._args_in = args_in + extra_post

        with captureStdOut() as cs:
            ret = self._func(*self._args_in)

        if not self._DEBUG:

            if self._sub:
                ret = 0
            else:
                # Special handling of returning a string
                if len(self._extra_pre):
                    ret = self._args[0].from_len(retstr, self._args_in[1])
                else:
                    ret = self._return.from_func(ret)

            dummy_args = {}
            count = 0

            for value, obj, ne in zip(
                    self._args_in[start:end], self._args[start:end], needs_extra[start:end]):
                if ne:
                    dummy_args[obj.name] = obj.from_len(
                        value, self._args_in[end])
                    end = end + 1
                else:
                    dummy_args[obj.name] = obj.from_func(value)
            if self._sub:
                ret = 0

            return Result(ret, dummy_args)
        else:
            return Result(ret, args_in)

    def _init_return(self):
        # make a object usable by _selectVar()
        ret = {}
        ret['var'] = self.proc['ret']
        self._return = self._get_fvar(ret)(ret)
        if isinstance(self._return, fStr):
            self._extra_pre = []
            # When returning a string we get a string and its len inserted at
            # the start of the argument list
            self._extra_pre.append(self._get_fvar(ret)(ret))
            self._extra_pre.append(fStrLen())

            self._args = self._extra_pre + self._args

    def _init_args(self):
        extras = []
        args = []
        for i in self.arg:
            x = self._get_fvar(i)(i)

            if isinstance(
                    x, fStr):  # Need a string length at the end of the argument list
                extras.append(fStrLen())
            args.append(x)

        args.extend(extras)
        return args

    def _get_fvar(self, var):
        return _selectVar(var)

    def _isSet(self, lib):
        pass


class fFuncPtr(fFunc):

    def _set_ptr_in_dll(self, lib, func):
        ff = getattr(lib, func.mangled_name)
        self.ctype()
        ptr = ctypes.c_void_p.in_dll(lib, self.mangled_name)
        ptr.value = ctypes.cast(ff, ctypes.c_void_p).value

        self._func = self._cfunc.in_dll(lib, self.mangled_name)

    def _isSet(self, lib):
        if not hasattr(self, '_cfunc'):
            raise AttributeError("Must point to something first")

        ptr = self._cfunc.in_dll(lib, self.mangled_name)
        if ctypes.cast(ptr, ctypes.c_void_p).value is None:
            raise AttributeError("Must point to something first")

    @property
    def ctype(self):
        if self._return is not None:
            self._cfunc = ctypes.CFUNCTYPE(self._return.ctype)
        else:
            self._cfunc = ctypes.CFUNCTYPE(None)
        return self._cfunc

    def in_dll(self, lib):
        self.lib = lib
        self._isSet(lib)

        return self

    def set_in_dll(self, lib, func):
        if not isinstance(func, fFunc):
            raise TypeError("Must be a fortran function")

        self._init_from_func(lib, func)
        self._set_ptr_in_dll(lib, func)

    def _init_from_func(self, lib, func):
        f = func.in_dll(lib)

        self._sub = f.proc['sub']
        self._args = f._args
        self._return = func._return
        self._extra_pre = f._extra_pre
        self._args_in = f._args_in

    def from_param(self, func):
        self.lib = func.lib
        self._init_from_func(self.lib, func)

        self.ctype()

        ff = getattr(self.lib, func.mangled_name)
        self.ctype()
        self.ptr = ctypes.c_void_p()
        self.ptr.value = ctypes.cast(ff, ctypes.c_void_p).value

        return self.ptr

    def from_func(self, pointer):
        return pointer

