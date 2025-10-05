# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

try:
    import pyquadp as pq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

from .base import f_type


class f_real(f_type):
    ftype = "real"
    default = 0.0


class f_real_4(f_real):
    kind = 4
    ctype = ctypes.c_float
    dtype = np.dtype("float32")


class f_real_8(f_real):
    kind = 8
    ctype = ctypes.c_double
    dtype = np.dtype("float64")


class f_real_16(f_real):
    kind = 16
    ctype = pq.c_qfloat
    dtype = np.dtype("B16")

    @property
    def value(self):
        return self._ctype.from_bytes(bytes(self._ctype.value))

    @value.setter
    def value(self, value):
        self._value = value
        self._ctype.value = self._ctype(value).to_bytes()

    @property
    def _as_parameter_(self):
        return self._ctype.to_bytes()
