# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .base import f_type


class f_logical(f_type):
    ftype = "logical"
    default = ".false."

    @property
    def value(self):
        self._value = bool(self._ctype.value)
        return self._value

    @value.setter
    def value(self, value):
        self._value = bool(value)
        if self._value:
            self._ctype.value = 1
        else:
            self._ctype.value = 0


class f_logical_1(f_logical):
    kind = 1
    ctype = ctypes.c_int8
    dtype = np.dtype(np.byte)


class f_logical_2(f_logical):
    kind = 2
    ctype = ctypes.c_int16
    dtype = np.dtype(np.short)


class f_logical_4(f_logical):
    kind = 4
    ctype = ctypes.c_int32
    dtype = np.dtype("int32")


class f_logical_8(f_logical):
    kind = 8
    ctype = ctypes.c_int64
    dtype = np.dtype("int64")
