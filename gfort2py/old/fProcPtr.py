# SPDX-License-Identifier: GPL-2.0+

import ctypes

from .fVar_t import fVar_t


class fProcPointer(fVar_t):
    def ctype(self):
        return self._func

    def from_param(self, param):
        # memmove?
        raise NotImplementedError

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        self.from_param(value)
