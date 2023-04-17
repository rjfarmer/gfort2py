# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fVar_t import fVar_t

from .fScalars import fScalar, fCmplx
from .fArrays import fExplicitArr, fAssumedShape, fAssumedSize
from .fStrings import fStr, fAllocStr
from .fDT import fDT, fExplicitDT
from .fProcPtr import fProcPointer


class fVar:
    def __new__(cls, obj, *args, **kwargs):
        if obj.is_derived():
            if obj.is_array():
                if obj.is_explicit():
                    return fExplicitDT(obj, fVar, *args, **kwargs)
                raise NotImplementedError
            else:
                return fDT(obj, fVar, *args, **kwargs)
        elif obj.is_proc_pointer():
            return fProcPointer(obj, *args, **kwargs)
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
                if obj.is_allocatable():
                    return fAllocStr(obj, *args, **kwargs)
                else:
                    return fStr(obj, *args, **kwargs)
            elif obj.is_complex():
                return fCmplx(obj, *args, **kwargs)
            else:
                return fScalar(obj, *args, **kwargs)
