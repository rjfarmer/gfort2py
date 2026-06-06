# SPDX-License-Identifier: GPL-2.0+

import ctypes

import numpy as np

try:
    import pyquadp as pyq  # type: ignore[import-not-found]

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

from .base import f_type


class ftype_real(f_type):
    ftype = "real"
    default = 0.0


class ftype_real_4(ftype_real):
    kind = 4
    ctype = ctypes.c_float
    dtype = np.dtype("float32")


class ftype_real_8(ftype_real):
    kind = 8
    ctype = ctypes.c_double
    dtype = np.dtype("float64")


class ftype_real_16(ftype_real):
    kind = 16

    @property
    def dtype(self) -> np.dtype:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.qarray.dtype

    @property
    def ctype(self):
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.c_qfloat

    @property
    def value(self) -> "pyq.qfloat":
        return pyq.qfloat.from_bytes(bytes(self._ctype))

    @value.setter
    def value(self, value: "pyq.qfloat"):
        if value is None:
            return

        self._value = pyq.qfloat(value)
        raw = self._value.to_bytes()
        ctypes.memmove(ctypes.addressof(self._ctype), raw, len(raw))

    @property
    def _as_parameter_(self):
        return self._ctype

    @property
    def pytype(self):
        return pyq.qfloat
