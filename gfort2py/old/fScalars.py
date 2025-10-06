# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .fVar_t import fVar_t

try:
    import pyquadp as pyq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False


class fScalar(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        if self.kind == 16:
            if PYQ_IMPORTED:
                p = pyq.qfloat(param).to_bytes()
            else:
                raise NotImplementedError(
                    f"Quad precision floats requires pyQuadp to be installed"
                )

            if self.cvalue is None:
                self.cvalue = self.ctype()()

            for i in range(16):
                self.cvalue[i] = p[i]

        else:
            if self.cvalue is None:
                self.cvalue = self.ctype()(param)
            else:
                self.cvalue.value = param

        return self.cvalue

    @property
    def value(self):
        if self.type == "INTEGER":
            return int(self.cvalue.value)
        elif self.type == "REAL":
            if self.kind == 16:
                if PYQ_IMPORTED:
                    return pyq.qfloat.from_bytes(bytes(self.cvalue))
                else:
                    raise TypeError(
                        f"Quad precision floats requires pyQuadp to be installed"
                    )
            elif self.kind == 8:
                return np.double(self.cvalue.value)
            else:
                return float(self.cvalue.value)
        elif self.type == "LOGICAL":
            return self.cvalue.value == 1

    @value.setter
    def value(self, value):
        self.from_param(value)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    @property
    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"


class fCmplx(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        if self.kind == 16:
            if PYQ_IMPORTED:
                p = pyq.qcmplx(param).to_bytes()
            else:
                raise NotImplementedError(
                    f"Quad precision complex requires pyQuadp to be installed"
                )

            if self.cvalue is None:
                self.cvalue = self.ctype()

            for i in range(32):
                self.cvalue[i] = p[i]

        else:
            self.cvalue.real = param.real
            self.cvalue.imag = param.imag

        return self.cvalue

    @property
    def value(self):
        x = self.cvalue

        if self.kind == 16:
            if PYQ_IMPORTED:
                return pyq.qcmplx.from_bytes(bytes(x))
            else:
                raise NotImplementedError(
                    f"Quad precision complex requires pyQuadp to be installed"
                )

        return complex(x.real, x.imag)

    @value.setter
    def value(self, value):
        self.from_param(value)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    @property
    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"
