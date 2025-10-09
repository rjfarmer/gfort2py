# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .base import f_type


class ftype_unsigned(f_type):
    ftype = "unsigned"
    default = 0


class ftype_unsigned_1(ftype_unsigned):
    kind = 1
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.ubyte)


class ftype_unsigned_2(ftype_unsigned):
    kind = 2
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.ushort)


class ftype_unsigned_4(ftype_unsigned):
    kind = 4
    ctype = ctypes.c_uint32
    dtype = np.dtype(np.uintc)


class ftype_unsigned_8(ftype_unsigned):
    kind = 8
    ctype = ctypes.c_uint64
    dtype = np.dtype(np.uint)
