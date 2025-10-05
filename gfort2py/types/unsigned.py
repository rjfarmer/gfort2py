# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .base import f_type


class f_unsigned(f_type):
    ftype = "unsigned"
    default = 0


class f_unsigned_1(f_unsigned):
    kind = 1
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.ubyte)


class f_unsigned_2(f_unsigned):
    kind = 2
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.ushort)


class f_unsigned_4(f_unsigned):
    kind = 4
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.uintc)


class f_unsigned_8(f_unsigned):
    kind = 8
    ctype = ctypes.c_uint64
    dtype = np.dtype(np.uint)
