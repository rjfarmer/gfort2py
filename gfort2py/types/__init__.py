# SPDX-License-Identifier: GPL-2.0+

import sys
from typing import Type
import abc

import gfModParser as gf

from .arrays import *
from .character import *
from .complex import *
from .dt import *
from .integer import *
from .logical import *
from .real import *
from .unsigned import *
from .module import *

__all__ = ["factory", "f_strlen", "f_optional", "get_module"]


def factory(obj: Type[gf.Symbol]) -> f_type:
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
                res = ftype_dt_explicit
            elif is_assumed_shape:
                res = ftype_dt_assumed_shape
            else:
                raise TypeError("Can't match object")
        else:
            res = ftype_dt
    elif is_array:
        if is_explicit:
            res = ftype_explicit_array
        elif is_assumed_shape:
            res = ftype_assumed_shape
        else:
            raise TypeError("Can't match object")

        # Inject in the base type
        def _base(self):
            cls = find_ftype(ftype, kind)

            def definition(self):
                return None

            # Inject into Fortran definition all the other module data
            cls.definition = classmethod(definition)

            abc.update_abstractmethods(cls)
            return cls()

        res._base = classmethod(_base)
    else:
        res = find_ftype(ftype, kind)

    def definition(self):
        self._obj = obj
        return self._obj

    # Inject into Fortran definition all the other module data
    res.definition = classmethod(definition)

    abc.update_abstractmethods(res)

    return res


def find_ftype(ftype, kind) -> f_type:
    name = f"ftype_{ftype}_{kind}"
    try:
        res = getattr(sys.modules[__name__], name)
    except Exception:
        raise TypeError("Can't match object")

    return res


class ftype_strlen(ftype_integer):
    @property
    def ctype(self):
        if is_64bit():
            return ctypes.c_int64
        else:
            return ctypes.c_int32


class ftype_optional(f_type):
    ctype = ctypes.c_byte
