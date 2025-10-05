# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .base import f_type


class f_integer(f_type):
    ftype = "integer"
    default = 0


class f_integer_1(f_integer):
    kind = 1
    ctype = ctypes.c_int8
    dtype = np.dtype(np.byte)


class f_integer_2(f_integer):
    kind = 2
    ctype = ctypes.c_int16
    dtype = np.dtype(np.short)


class f_integer_4(f_integer):
    kind = 4
    ctype = ctypes.c_int32
    dtype = np.dtype(">i4")


class f_integer_8(f_integer):
    kind = 8
    ctype = ctypes.c_int64
    dtype = np.dtype(">i8")


class f_optional(f_type):
    ctype = ctypes.c_byte
