# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np

from .fVar_t import fVar_t

from .utils import copy_array


_all_dts = {}


def make_dt(name):
    class _fDerivedType(ctypes.Structure):
        pass

    _fDerivedType.__name__ = name
    return _fDerivedType


class fDT(fVar_t):
    def __init__(self, obj, fvar, allobjs=None, cvalue=None):
        self.obj = obj
        self.fvar = fvar
        self.allobjs = allobjs
        self.cvalue = cvalue
        self.unpack = True

        # Get obj for derived type spec
        self._dt_obj = self.allobjs[self.obj.dt_type()]
        self._dt_name = self._dt_obj.name

        self._init_args()

        self._ctype = None

    def _init_args(self):
        # Store fVar's for each component
        self._dt_args = {}
        for var in self._dt_obj.dt_components():
            # Catch dt's which contain themselves:
            if var.is_derived() and var.dt_type() == self.obj.dt_type():
                raise NotImplementedError(
                    "Derived types containing themselves not supported yet"
                )

            self._dt_args[var.name] = self.fvar(var, allobjs=self.allobjs)

    def ctype(self):
        if self._ctype is None:
            fields = []
            for var in self._dt_obj.dt_components():
                fields.append((var.name, self._dt_args[var.name].ctype()))

            class _fDerivedType(ctypes.Structure):
                _fields_ = fields

            if self._dt_name not in _all_dts:
                _all_dts[self._dt_name] = _fDerivedType
            else:
                try:
                    _all_dts[self._dt_name]._fields_ = fields
                except AttributeError:
                    pass

            self._ctype = _all_dts[self._dt_name]

        return self._ctype

    def from_ctype(self, ct):
        self.cvalue = ct
        return self.value

    def from_address(self, addr):
        self.cvalue = self.ctype().from_address(addr)
        return self.cvalue

    def in_dll(self, lib):
        self.cvalue = self.ctype().in_dll(lib, self.mangled_name)
        return self.cvalue

    def from_param(self, param):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        for key, value in param.items():
            if key not in self._dt_args:
                raise KeyError(f"{key} not present in {self._dt_obj.name}")

            if self._dt_args[key].obj.is_derived():
                self._dt_args[key] = self.fvar(
                    self.allobjs[key],
                    allobjs=self.allobjs,
                    cvalue=getattr(self.cvalue, key),
                )
                for k, v in value.items():
                    self._dt_args[key].__setitem__(k, v)
            else:
                if not isinstance(value, fVar_t):
                    self._dt_args[key].from_param(value)
                else:
                    self._dt_args[key].from_ctype(value.cvalue())

                v = self._dt_args[key].cvalue

                if self._dt_args[key].is_array:
                    copy_array(
                        ctypes.addressof(v),
                        ctypes.addressof(getattr(self.cvalue, key)),
                        1,
                        ctypes.sizeof(v),
                    )
                else:
                    setattr(self.cvalue, key, v)

        return self.cvalue

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        self.from_param(value)

    def keys(self):
        return [i.name for i in self._dt_obj.dt_components()]

    def values(self):
        return [self.__getitem__(key) for key in self.keys()]

    def items(self):
        return [(key, self.__getitem__(key)) for key in self.keys()]

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise AttributeError("Scalar derived type can't be sliced")

        if key in self._dt_args:
            return self._dt_args[key].from_ctype(getattr(self.cvalue, key))
        else:
            raise KeyError(f"{key} not present in {self._dt_obj.name}")

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise AttributeError("Scalar derived type can't be sliced")

        if key in self._dt_args:
            self.from_param({key: value})
        else:
            raise KeyError(f"{key} not present in {self._dt_obj.name}")

    def __dir__(self):
        return list(self._dt_args.keys())

    def __getattr__(self, key):
        if "_dt_args" in self.__dict__:
            if key in self._dt_args:
                raise AttributeError("Can't get components as attributes")

        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        if "_dt_args" in self.__dict__:
            if key in self._dt_args:
                raise AttributeError("Can't set components as attributes")

        if key == "value":
            self.from_param(value)

        self.__dict__[key] = value


class fExplicitDT(fVar_t):
    def __init__(self, obj, fvar, allobjs=None, cvalue=None):
        self.obj = obj
        self.fvar = fvar
        self.allobjs = allobjs
        self.cvalue = cvalue
        self.unpack = True

        # Get obj for derived type spec
        self._dt_obj = self.allobjs[self.obj.dt_type()]

        self._dt_ctype = fDT(self.obj, self.fvar, allobjs=self.allobjs)
        self._saved = {}

    def ctype(self):
        self._ctype = self._dt_ctype.ctype() * self.obj.size
        return self._ctype

    def __getitem__(self, index):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        if isinstance(index, tuple):
            ind = np.ravel_multi_index(index, self.obj.shape(), order="F")
        else:
            ind = index

        if ind > self.obj.size:
            raise IndexError("Out of bounds")

        if ind not in self._saved:
            self._saved[ind] = fDT(
                self.obj, self.fvar, allobjs=self.allobjs, cvalue=self.cvalue[ind]
            )

        return self._saved[ind]

    def __setitem__(self, index, value):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        if isinstance(index, tuple):
            ind = np.ravel_multi_index(index, self.obj.shape(), order="F")
        else:
            ind = index

        if ind > self.obj.size:
            raise IndexError("Out of bounds")

        if ind not in self._saved:
            self._saved[ind] = fDT(
                self.obj, allobjs=self.allobjs, cvalue=self.cvalue[ind]
            )

        self._saved[ind].value = value

    def from_param(self, param):
        if self.cvalue is None:
            self.cvalue = self.ctype()()

        for index, value in enumerate(param):
            self.__setitem__(index, value)

        return self.cvalue

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        self.from_param(value)
