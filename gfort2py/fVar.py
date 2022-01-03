# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fVar_t import *

class fVar:
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._lib = lib

        self._value = fVar_t(self._allobjs[key])

    def from_param(self, value):
        return self._value.from_param(value)

    @property
    def value(self):
        return self._value.from_ctype(self._in_dll(self._lib))

    @value.setter
    def value(self, value):
        ct = self._in_dll(self._lib)
        self._value.set_ctype(ct, value)

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def _in_dll(self, lib):
        return self._value.ctype().in_dll(lib, self._value.mangled_name())

    @property
    def module(self):
        return self._value.module



class fParam:
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._obj = self._allobjs[key]
        self._lib = lib

    @property
    def value(self):
        return self._obj.value()

    @value.setter
    def value(self, value):
        raise AttributeError("Parameters can't be altered")