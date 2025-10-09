# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod

try:
    import pyquadp as pq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

from .base import f_type


class ftype_complex(f_type, metaclass=ABCMeta):
    ftype = "complex"
    default = 0

    @property
    @abstractmethod
    def base_ctype(self):
        raise NotImplementedError

    @property
    def ctype(self):
        class cmplx(ctypes.Structure):
            _fields_ = [
                ("real", self.base_ctype),
                ("imag", self.base_ctype),
            ]

        return cmplx()

    @property
    def value(self):
        self._value = complex(self._ctype.value.real, self._ctype.value.imag)
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._ctype.real.value = self._value.real
        self._ctype.imag.value = self._value.imag


class ftype_complex_4(ftype_complex):
    base_ctype = ctypes.c_float
    kind = 4
    dtype = np.dtype(np.csingle)


class ftype_complex_8(ftype_complex):
    base_ctype = ctypes.c_double
    kind = 8
    dtype = np.dtype(np.cdouble)


class ftype_complex_16(ftype_complex):
    ctype = pq.c_qcmplx
    kind = 16
    dtype = np.dtype("S32")

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
