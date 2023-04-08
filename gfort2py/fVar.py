# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fVar_t import *


class fVar:
    def __init__(self, obj, allobjs, _cvalue=None):
        self.obj = obj
        self._cvalue = _cvalue

        if self.obj.is_derived():
            self._value = fDT(self.obj, allobjs, self._cvalue)
            # TODO: Handle arrays
        elif self.obj.is_array():
            if self.obj.is_explicit():
                self._value = fExplicitArr(self.obj, allobjs, self._cvalue)
            elif self.obj.is_assumed_size():
                self._value = fAssumedSize(self.obj, allobjs, self._cvalue)
            elif (
                self.obj.is_assumed_shape()
                or self.obj.is_allocatable()
                or self.obj.is_pointer()
            ):
                self._value = fAssumedShape(self.obj, allobjs, self._cvalue)
            else:
                raise TypeError("Unknown array type")
        else:
            if self.obj.is_char():
                self._value = fStr(self.obj, allobjs, self._cvalue)
            elif self.obj.is_complex():
                self._value = fCmplx(self.obj, allobjs, self._cvalue)
            else:
                self._value = fScalar(self.obj, allobjs, self._cvalue)

    def from_param(self, value):
        return self._value.from_param(value)

    def ctype(self):
        return self._value.ctype()

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, value):
        self._value.value = value

    def __repr__(self):
        return repr(self._value)

    def __str__(self):
        return str(self.value)

    def in_dll(self, lib):
        return self._value.in_dll(lib)

    @property
    def module(self):
        return self._value.module

    def from_address(self, addr):
        return self._value.from_address(addr)

    def __doc__(self):
        return self._value.__doc__()

    @property
    def name(self):
        return self._value.name

    def ctype_len(self):
        return self._value.ctype_len()

    def len(self):
        return self._value.len()

    def from_ctype(self, ct):
        return self._value.from_ctype(ct)

    def keys(self):
        if not self.obj.is_derived():
            raise NotImplemented
        return self._value.keys()

    def items(self):
        if not self.obj.is_derived():
            raise NotImplemented
        return self._value.items()

    def values(self):
        if not self.obj.is_derived():
            raise NotImplemented
        return self._value.values()

    def __getitem__(self, key):
        if not self.obj.is_derived():
            raise NotImplemented
        return self._value.__getitem__(key)

    def __setitem__(self, key, value):
        if not self.obj.is_derived():
            raise NotImplemented
        return self._value.__setitem__(key, value)


class fParam:
    def __init__(self, obj):
        self.obj = obj

    @property
    def value(self):
        return self.obj.value()

    @value.setter
    def value(self, value):
        raise AttributeError("Parameters can't be altered")

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    @property
    def module(self):
        return self._value.module
