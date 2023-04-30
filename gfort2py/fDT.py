import ctypes
import numpy as np

from .fVar_t import fVar_t


class fDT(fVar_t):
    def __init__(self, obj, fvar, allobjs=None, cvalue=None):
        self.obj = obj
        self.fvar = fvar
        self.allobjs = allobjs
        self.cvalue = cvalue

        # Get obj for derived type spec
        self._dt_obj = self.allobjs[self.obj.dt_type()]

        self._init_args()

        self._ctype = None

    def _init_args(self):
        # Store fVar's for each component
        self._dt_args = {}
        if not any([var.is_derived() for var in self._dt_obj.dt_components()]):
            for var in self._dt_obj.dt_components():
                self._dt_args[var.name] = self.fvar(var, allobjs=self.allobjs)
        else:
            raise NotImplementedError

    def ctype(self):
        if self._ctype is None:
            # See if this is a "simple" (no other dt's) dt
            if not any([var.is_derived() for var in self._dt_obj.dt_components()]):
                fields = []
                for var in self._dt_obj.dt_components():
                    # print(var.name,self._dt_args[var.name].ctype())
                    fields.append((var.name, self._dt_args[var.name].ctype()))

                class _fDerivedType(ctypes.Structure):
                    _fields_ = fields

            else:
                raise NotImplementedError

            self._ctype = _fDerivedType

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
            # print(key,value)
            if key not in self._dt_args:
                raise KeyError(f"{key} not present in {self._dt_obj.name}")

            if not isinstance(value, fVar_t):
                self._dt_args[key].from_param(value)
            else:
                self._dt_args[key].from_ctype(value.cvalue())

            v = self._dt_args[key].cvalue

            # print(self.cvalue)
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
        if key in self._dt_args:
            return self._dt_args[key].from_ctype(getattr(self.cvalue, key))
        else:
            raise KeyError(f"{key} not present in {self._dt_obj.name}")

    def __setitem__(self, key, value):
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
