# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from .fVar_t import fVar_t


class fScalar(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
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
                raise NotImplementedError(f"Quad precision floats not supported yet")
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

    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"


class fCmplx(fVar_t):
    def ctype(self):
        return self._ctype_base

    def from_param(self, param):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        self.cvalue.real = param.real
        self.cvalue.imag = param.imag
        return self.cvalue

    @property
    def value(self):
        x = self.cvalue

        if self.kind == 16:
            raise NotImplementedError(
                f"Quad precision complex numbers not supported yet"
            )
        return complex(x.real, x.imag)

    @value.setter
    def value(self, value):
        self.from_param(value)

    def sizeof(self):
        return ctypes.sizeof(self.ctype)

    def __doc__(self):
        return f"{self.type}(KIND={self.kind}) :: {self.name}"
