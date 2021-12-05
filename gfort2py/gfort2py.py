# SPDX-License-Identifier: GPL-2.0+
import ctypes
import numpy as np
from collections import namedtuple

from . import parseMod as pm

class fObject:
    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __add__(self, other):
        return self.value.__add__(other)

    def __sub__(self, other):
        return self.value.__sub__(other)

    def __matmul__(self, other):
        return self.value.__matmul__(other)

    def __truediv__(self, other):
        return self.value.__truediv__(other)

    def __floordiv__(self, other):
        return self.value.__floordiv__(other)

    def __mod__(self, other):
        return self.value.__mod__(other)

    def __divmod__(self, other):
        return self.value.__divmod__(other)

    def __pow__(self, other, modulo=None):
        return self.value.__pow__(other, modulo)

    def __lshift__(self, other):
        return self.value.__lshift__(other)

    def __rshift__(self, other):
        return self.value.__rshift__(other)

    def __and__(self, other):
        return self.value.__and__(other)

    def __xor__(self, other):
        return self.value.__xor__(other)

    def __or__(self, other):
        return self.value.__or__(other)

    def __radd__(self, other):
        return self.value.__radd__(other)

    def __rsub__(self, other):
        return self.value.__rsub__(other)

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(other)

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(other)

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(other)

    def __rmod__(self, other):
        return self.value.__rmod__(other)

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(other)

    def __rlshift__(self, other):
        return self.value.__rlshift__(other)

    def __rrshift__(self, other):
        return self.value.__rrshift__(other)

    def __rand__(self, other):
        return self.value.__rand__(other)

    def __rxor__(self, other):
        return self.value.__rxor__(other)

    def __ror__(self, other):
        return self.value.__ror__(other)

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __complex__(self):
        return self.value.__complex__()

    def __int__(self):
        return self.value.__int__()

    def __float__(self):
        return self.value.__float__()

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)

    def __trunc__(self):
        return self.value.__trunc__()

    def __floor__(self):
        return self.value.__floor__()

    def __ceil__(self):
        return self.value.__ceil__()


class fParam(fObject):
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

    @property
    def value(self):
        return self._object.sym.parameter.value

    @value.setter
    def value(self, value):
        raise AttributeError("Parameter can't be altered")



class fVar_t:
    def __init__(self, obj):
        self._object = obj

    def type(self):
        return self._object.sym.ts.type

    def flavor(self):
        return self._object.sym.flavor

    def kind(self):
        return self._object.sym.ts.kind

    def from_param(self, value):
        return self.ctype(value)

    def is_pointer(self):
        return 'POINTER' in self._object.sym.attr.attributes

    def is_value(self):
        return 'VALUE' in self._object.sym.attr.attributes

    def is_optional(self):
        return 'OPTIONAL' in self._object.sym.attr.attributes

    @property
    def ctype(self):
        t = self.type()
        k = int(self.kind())

        if t == 'INTEGER':
            if k == 4:
                return ctypes.c_int32
            elif k == 8:
                return ctypes.c_int64
        elif t == 'REAL':
            if k == 4:
                return ctypes.c_float
            elif k == 8:
                return ctypes.c_double
        elif t == 'LOGICAL':
            return ctypes.c_int32

        raise NotImplementedError(f'Object of type {t} and kind {k} not supported yet')

    @property
    def name(self):
        return self._object.head.name

    @property
    def __doc__(self):
        return f"{self._object.head.name} {self._value.type()}(KIND={self._value.kind()})"

class fVar(fObject):
    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

        self._value = fVar_t(self._object)

    def from_param(self, value):
        return self.ctype(value)

    @property
    def value(self):
        return self.in_dll(self._lib).value

    @value.setter
    def value(self, value):
        self.in_dll(self._lib).value = value

    @property
    def mangled_name(self):
        return self._object.head.mn_name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def in_dll(self, lib):
        return self._value.ctype.in_dll(lib, self.mangled_name)

    @property
    def module(self):
        return self._object.head.module

    @property
    def __doc__(self):
        return f"{self._value._type()}(KIND={self._value._kind()}) " \
               f"MODULE={self.module}.mod"


class fProc:
    Result = namedtuple('Result', ["res", "args"])


    def __init__(self, lib, allobjs, key):
        self._allobjs = allobjs
        self._object = self._allobjs[key]
        self._lib = lib

        self._func = getattr(lib, self.mangled_name)
        self._set_return()
        self._set_argtypes()

    @property
    def mangled_name(self):
        return self._object.head.mn_name

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def in_dll(self, lib):
        return self._func

    @property
    def module(self):
        return self._object.head.module

    @property
    def __doc__(self):
        return f"Procedure"

    def __call__(self, *args, **kwargs):

        func_args = self._convert_args(*args, **kwargs)

        if func_args is not None:
            res = self._func(*func_args)
        else:
            res = self._func()

        return self._convert_result(res, func_args)

    def _set_return(self):
        symref = self._object.sym.sym_ref.ref

        if symref == 0:
            self._func.restype = None # Subroutine
        else:
            self._func.restype = fVar_t(self._allobjs[symref]).ctype

    def _set_argtypes(self):
        fargs = self._object.sym.formal_arg

        res = []
        if not len(fargs):
            return

        for i in fargs:
            var = fVar_t(self._allobjs[i.ref])

            if var.is_value():
                a = var.ctype
            elif var.is_pointer():
                a = ctypes.POINTER(ctypes.POINTER(var.ctype))
            else:
                a = ctypes.POINTER(var.ctype)

            res.append(a)

        self._func.argtypes = res


    def _convert_args(self, *args, **kwargs):
        fargs = self._object.sym.formal_arg

        res = []
        if not len(args):
            return None

        for value,fval in zip(args,fargs):
            var = fVar_t(self._allobjs[fval.ref])
            z = var.ctype(value)

            if var.is_value():
                a = z
            elif var.is_pointer():
                a = ctypes.pointer(ctypes.pointer(z))
            else:
                a = ctypes.pointer(z)

            res.append(a)

        return res

    def _convert_result(self, result, args):
        fargs = self._object.sym.formal_arg
        res = {}

        if len(fargs):
            for ptr,fval in zip(args,fargs):
                x = ptr
                if hasattr(ptr, "contents"):
                    if hasattr(ptr.contents, "contents"):
                        x = ptr.contents.contents
                    else:
                        x = ptr.contents

                if hasattr(x, "value"):
                    x = x.value

                res[self._allobjs[fval.ref].head.name] = x

        return self.Result(result, res)

class fFort:
    _initialised = False

    def __init__(self, libname, mod_file):
        self._lib = ctypes.CDLL(libname)
        self._mod_file = mod_file 
        self._module = pm.module(self._mod_file)

        self._initialised = True

    def keys(self):
        return self._module.keys()

    def __contains__(self, key):
        return key in self._module.keys()

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]

        if '_initialised' in self.__dict__:
            if self._initialised:
                if key not in self.keys():
                    raise AttributeError(f"{self._mod_file}  has no attribute {key}")

            flavor = self._module[key].sym.attr.flavor
            if flavor == 'VARIABLE':
                return fVar(self._lib, self._module, key)
            elif flavor == 'PROCEDURE':
                return fProc(self._lib, self._module, key)
            elif flavor == 'PARAMETER':
                return fParam(self._lib, self._module, key)
            else:
                raise NotImplementedError(f"Object type {flavor} not implemented yet")


    def __setattr__(self, key, value):
        if '_initialised' in self.__dict__:
            if self._initialised:
                if key not in self:
                    raise AttributeError(f"{self._mod_file}  has no attribute {key}")

                flavor = self._module[key].sym.attr.flavor
                if flavor == 'VARIABLE':
                    f = fVar(self._lib, self._module, key)
                    f.value = value
                    return
                elif flavor == 'PARAMETER':
                    raise AttributeError('Can not alter a parameter')

                else:
                    raise NotImplementedError(f"Object type {flavor} not implemented yet")

        self.__dict__[key] = value

    @property
    def __doc__(self):
        return f"MODULE={self._module.filename}"

    def __str__(self):
        return f"{self._module.filename}"
