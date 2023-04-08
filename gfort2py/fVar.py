# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fVar_t import *


class fVar:
    def __new__(cls, obj, *args, **kwargs):
        if obj.is_derived():
            return fDT(obj, *args, **kwargs)
            # TODO: Handle arrays
        elif obj.is_array():
            if obj.is_explicit():
                return fExplicitArr(obj, *args, **kwargs)
            elif obj.is_assumed_size():
                return fAssumedSize(obj, *args, **kwargs)
            elif obj.is_assumed_shape() or obj.is_allocatable() or obj.is_pointer():
                return fAssumedShape(obj, *args, **kwargs)
            else:
                raise TypeError("Unknown array type")
        else:
            if obj.is_char():
                return fStr(obj, *args, **kwargs)
            elif obj.is_complex():
                return fCmplx(obj, *args, **kwargs)
            else:
                return fScalar(obj, *args, **kwargs)


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
