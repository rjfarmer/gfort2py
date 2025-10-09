# SPDX-License-Identifier: GPL-2.0+

import ctypes
import numpy as np

from ..allocate import allocate_dt

from .base import f_type
from .arrays import ftype_assumed_shape, ftype_explicit_array

__all__ = ["ftype_dt", "ftype_dt_explicit", "ftype_dt_assumed_shape"]


_all_dts = {}


class ftype_dt(f_type):
    dtype = None
    kind = -1

    def __init__(self, *, ftype: str, fields: dict[str, f_type]):
        self._ftype = ftype
        self.fields = fields
        super().__init__()

    @property
    def ftype(self):
        return self._ftype

    @property
    def ctype(self):
        if self.ftype not in _all_dts:

            class dt(ctypes.Structure):
                _fields_ = list(self.fields.items())

            _all_dts[self.ftype] = dt

        return _all_dts[self.ftype]

    def __getitem__(self, key):
        return getattr(self._ctype, key)

    def __setitem__(self, key, value):
        setattr(self._ctype, key, value)

    @property
    def value(self):
        res = {}
        for key in self.fields.keys():
            res[key] = self[key]

    @value.setter
    def value(self, value):
        self._value = value
        for key, value in value.items():
            self[key] = value

    def keys(self):
        return self.fields.keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        return f"type({self.ftype})"


class ftype_dt_array(f_type):
    def __init__(self, ftype, fields, shape, array_cls):
        self.ftype = ftype
        self.fields = fields
        self.shape = shape
        self._array_cls = array_cls
        self._dt = f_dt(ftype=self.ftype, fields=self.fields)
        super().__init__()

    @property
    def ctype(self):
        return self._array_cls(self._dt, shape=self.shape)

    def _index(self, index):
        if isinstance(index, tuple):
            ind = np.ravel_multi_index(index, self.shape(), order="F")
        else:
            ind = index

        if ind > np.prod(self.shape):
            raise IndexError("Out of bounds")

        return ind

    def __getitem__(self, key):
        ind = self._index(key)
        new_dt = f_dt(ftype=self.ftype, fields=self.fields)
        new_dt._ctype = self._ctype[ind]
        return new_dt

    def __setitem__(self, key, value):
        ind = self._index(key)
        self._ctype[ind][key] = value

    def keys(self):
        return self.fields.keys()

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    def __dir__(self):
        return list(self.keys())


class ftype_dt_explicit(ftype_dt_array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, array_cls=f_explicit_array)

    def __repr__(self):
        return f"type({self.ftype})({self.shape})"


class ftype_dt_assumed_shape(ftype_dt_array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, array_cls=_f_dt_assumed_shape)

    def __repr__(self):
        s = ",".join([i for i in self.shape])
        return f"type({self.ftype})({s})"


class _f_dt_assumed_shape(ftype_assumed_shape):
    def _allocate(self, shape):
        allocate_dt(
            self.ctype, type=self._base.ftype, shape=shape, module="", library=""
        )
