# SPDX-License-Identifier: GPL-2.0+

import ctypes
import collections

from .utils import resolve_other_args

try:
    import pyquadp as pyq

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False


class fVar_t:
    """
    Base class for all objects, everything should inheriet from this
    or another class that evntauly inherits from this.

    """

    Args = collections.namedtuple("arg", ["prepend", "arg", "append"])

    def __init__(self, obj, allobjs=None, cvalue=None):
        self.obj = obj
        self.allobjs = allobjs
        self.cvalue = cvalue
        self.unpack = True
        self.is_array = False

        self.type, self.kind = self.obj.type_kind()

        self._ctype_base = ctype_map(self.type, self.kind)

    @property
    def name(self):
        return self.obj.name

    @property
    def mangled_name(self):
        return self.obj.mangled_name

    @property
    def module(self):
        return self.obj.module

    def ctype_len(self, *args):
        return None

    def from_ctype(self, ct):
        if hasattr(ct, "__ctypes_from_outparam__"):
            self.cvalue = ct
        else:  # Not actually a ctype
            self.from_param(ct)
            # Do it this way so we get the conversion code in value called (i.e decodeing bytes to a string)
        return self.value

    def from_address(self, addr):
        self.cvalue = self.ctype().from_address(addr)
        return self.cvalue

    def in_dll(self, lib):
        self.cvalue = self.ctype().in_dll(lib, self.mangled_name)
        return self.cvalue

    def to_proc(self, value, other_args):
        start = None
        arg = None
        end = None

        if self.obj.is_optional() and value is None and not self.obj.is_char():
            end = ctypes.c_byte(0)
            arg = None
            return self.Args(start, arg, end)

        # optional characters dont have a byte at the end just a NULL pointer to the char
        if self.obj.is_optional() and value is None and self.obj.is_char():
            end = ctypes.c_int64(0)
            arg = None
            return self.Args(start, arg, end)

        raw_arg = self.from_param(value)
        if self.obj.is_optional() and not self.obj.is_char():
            end = ctypes.c_byte(1)

        if self.obj.is_value():
            arg = raw_arg
        else:
            if self.obj.is_pointer():
                if self.obj.not_a_pointer():
                    arg = ctypes.pointer(raw_arg)
                else:
                    arg = ctypes.pointer(ctypes.pointer(raw_arg))
            else:
                arg = ctypes.pointer(raw_arg)

        if self.obj.is_deferred_len() or self.obj.is_char():
            end = self.ctype_len(value)

        return self.Args(start, arg, end)


def ctype_map(type, kind):
    if type == "INTEGER":
        if kind == 1:
            return ctypes.c_int8
        elif kind == 2:
            return ctypes.c_int16
        if kind == 4:
            return ctypes.c_int32
        elif kind == 8:
            return ctypes.c_int64
        elif kind == 16:
            if PYQ_IMPORTED:
                return ctypes.c_byte * 16
            else:
                raise TypeError(f"Quad precision ints requires pyQuadp to be installed")
        else:
            raise TypeError("Integer type of kind={kind} not supported")
    elif type == "REAL":
        if kind == 4:
            return ctypes.c_float
        elif kind == 8:
            return ctypes.c_double
        elif kind == 16:
            if PYQ_IMPORTED:
                return ctypes.c_byte * 16
            else:
                raise TypeError(
                    f"Quad precision floats requires pyQuadp to be installed"
                )
        else:
            raise TypeError("Float type of kind={kind} not supported")
    elif type == "LOGICAL":
        return ctypes.c_int32
    elif type == "CHARACTER":
        return ctypes.c_char
    elif type == "COMPLEX":
        if kind == 4:
            ct = ctypes.c_float
        elif kind == 8:
            ct = ctypes.c_double
        elif kind == 16:
            if PYQ_IMPORTED:
                return ctypes.c_byte * 32
            else:
                raise TypeError(
                    f"Quad precision complex requires pyQuadp to be installed"
                )
        else:
            raise TypeError("Complex type of kind={kind} not supported")

        class complex(ctypes.Structure):
            _fields_ = [
                ("real", ct),
                ("imag", ct),
            ]

        return complex
    else:
        raise TypeError(f"Type={type} and kind={kind} not supported")
