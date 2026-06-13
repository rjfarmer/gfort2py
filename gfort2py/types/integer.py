# SPDX-License-Identifier: GPL-2.0+

import ctypes

import numpy as np

try:
    import pyquadp as pyq  # type: ignore[import-not-found]

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

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
    dtype = np.dtype("i4")


class ftype_integer_8(ftype_integer):
    kind = 8
    ctype = ctypes.c_int64
    dtype = np.dtype("i8")


class ftype_integer_16(ftype_integer):
    kind = 16

    @property
    def dtype(self) -> np.dtype:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.qiarray.dtype

    @property
    def ctype(self):
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.c_qint

    @property
    def value(self) -> "pyq.qint":
        return pyq.qint.from_bytes(bytes(self._ctype))

    @value.setter
    def value(self, value: "pyq.qint"):
        if value is None:
            return

        self._value = pyq.qint(value)
        raw = self._value.to_bytes()
        ctypes.memmove(ctypes.addressof(self._ctype), raw, len(raw))

    @property
    def _as_parameter_(self):
        return self._ctype

    @property
    def pytype(self):
        return pyq.qint
