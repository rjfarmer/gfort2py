# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .base import f_type


class ftype_integer(f_type):
    ftype = "integer"
    default = 0


class ftype_integer_1(ftype_integer):
    kind = 1
    ctype = ctypes.c_int8
    dtype = np.dtype(np.byte)


class ftype_integer_2(ftype_integer):
    kind = 2
    ctype = ctypes.c_int16
    dtype = np.dtype(np.short)


class ftype_integer_4(ftype_integer):
    kind = 4
    ctype = ctypes.c_int32
    dtype = np.dtype(">i4")


class ftype_integer_8(ftype_integer):
    kind = 8
    ctype = ctypes.c_int64
    dtype = np.dtype(">i8")
