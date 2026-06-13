# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta, abstractmethod

import numpy as np

try:
    import pyquadp as pyq  # type: ignore[import-not-found]

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

        return cmplx

    @property
    def value(self) -> complex:
        self._value = complex(self._ctype.real, self._ctype.imag)
        return self._value

    @value.setter
    def value(self, value: complex):
        if value is None:
            return

        self._value = value
        self._ctype.real = self._value.real
        self._ctype.imag = self._value.imag


class ftype_complex_4(ftype_complex):
    base_ctype = ctypes.c_float
    kind = 4
    dtype = np.dtype(np.csingle)


class ftype_complex_8(ftype_complex):
    base_ctype = ctypes.c_double
    kind = 8
    dtype = np.dtype(np.cdouble)


class ftype_complex_16(ftype_complex):
    kind = 16

    @property
    def dtype(self) -> np.dtype:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.qcarray.dtype

    @property
    def base_ctype(self):
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.c_qfloat

    @property
    def ctype(self):
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")
        return pyq.c_qcmplx

    @property  # type: ignore[override]
    def value(self) -> "pyq.qcmplx":
        return pyq.qcmplx.from_bytes(bytes(self._ctype))

    @value.setter
    def value(self, value: "pyq.qcmplx"):
        if value is None:
            return

        self._value = pyq.qcmplx(value)
        raw = self._value.to_bytes()
        ctypes.memmove(ctypes.addressof(self._ctype), raw, len(raw))

    @property
    def _as_parameter_(self):
        return self._ctype

    @property
    def pytype(self):
        return pyq.qcmplx
