# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type

import gfModParser as gf

from .arrays import *
from .character import *
from .complex import *
from .dt import *
from .integer import *
from .logical import *
from .real import *
from .unsigned import *

__all__ = ["factory", "f_strlen", "f_optional"]


def factory(obj: Type[gf.Symbol]):
    """Factory class to convert a (ftype,kind) into a wrapper object

    Args:
        obj: A single Variable from gfModParser
    Raises:
        TypeError: If type is unparsable raise TypeError

    Returns:
        f_type: A wrapper object for converting a Python type into/out of a ctype compatible with Fortran.
    """

    ftype = obj.type.lower()
    kind = obj.kind
    is_array = obj.is_array
    is_dt = obj.is_dt
    is_explicit = obj.properties.array_spec.is_explicit
    is_assumed_shape = obj.properties.array_spec.is_deferred

    if is_dt:
        if is_array:
            if is_explicit:
                return f_dt_explicit
            elif is_assumed_shape:
                return f_dt_assumed_shape
            raise TypeError("Can't match object")
        else:
            return f_dt
    elif is_array:
        if is_explicit:
            return f_explicit_array
        elif is_assumed_shape:
            return f_assumed_shape
        raise TypeError("Can't match object")

    else:
        name = f"f_{ftype}_{kind}"
        try:
            return getattr(sys.modules[__name__], name)
        except Exception:
            raise TypeError("Can't match object")


class f_strlen(f_integer):
    @property
    def ctype(self):
        if is_64bit():
            return ctypes.c_int64
        else:
            return ctypes.c_int32


class f_optional(f_type):
    ctype = ctypes.c_byte


def factory_init(ctype, obj, module):
    """Initializes ctype, taking into account extra args needed"""
    pass
