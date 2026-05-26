# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf

from .arrays import *
from .base import AllocStrategy, FortranSymbol, f_type
from .character import *
from .complex import *
from .dt import *
from .integer import *
from .logical import *
from .module import *
from .parameters import *
from .real import *
from .unsigned import *

__all__ = [
    "factory",
    "f_strlen",
    "f_optional",
    "get_module",
    "fParam",
    "FortranSymbol",
    "AllocStrategy",
    "register_ftype",
]

# Registry mapping (ftype, kind) -> f_type subclass.
# Classes stored here may still have abstract methods (e.g. `definition`) that are
# injected at runtime by the factory, so the value type is type[Any].
# Use register_ftype() to add entries; kind=None means "any kind" (e.g. character).
_ftype_registry: dict[tuple[str, int | None], type[Any]] = {
    ("integer", 1): ftype_integer_1,
    ("integer", 2): ftype_integer_2,
    ("integer", 4): ftype_integer_4,
    ("integer", 8): ftype_integer_8,
    ("real", 4): ftype_real_4,
    ("real", 8): ftype_real_8,
    ("real", 16): ftype_real_16,
    ("complex", 4): ftype_complex_4,
    ("complex", 8): ftype_complex_8,
    ("complex", 16): ftype_complex_16,
    ("logical", 1): ftype_logical_1,
    ("logical", 2): ftype_logical_2,
    ("logical", 4): ftype_logical_4,
    ("logical", 8): ftype_logical_8,
    ("unsigned", 1): ftype_unsigned_1,
    ("unsigned", 2): ftype_unsigned_2,
    ("unsigned", 4): ftype_unsigned_4,
    ("unsigned", 8): ftype_unsigned_8,
    # character: matched by ftype alone (kind=None)
    ("character", None): ftype_character,
}


def register_ftype(ftype: str, kind: int | None, cls: type[Any]) -> None:
    """Register a concrete f_type subclass for a given Fortran (ftype, kind) pair.

    Args:
        ftype: Fortran type name in lowercase (e.g. ``"integer"``).
        kind: Byte size of the kind, or ``None`` for types matched by ftype alone
              (e.g. ``"character"``).
        cls: The :class:`f_type` subclass to associate with this pair.
    """
    _ftype_registry[(ftype, kind)] = cls


def factory(obj: gf.Symbol) -> type[f_type]:
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
    is_assumed_rank = obj.properties.array_spec.is_assumed_rank

    if is_dt:
        if is_array:
            if is_explicit:
                res: Any = ftype_dt_explicit
            elif is_assumed_shape:
                res = ftype_dt_assumed_shape
            else:
                raise TypeError("Can't match object")
        else:
            res = ftype_dt
    elif is_array:
        base_arr_cls: Any
        if is_explicit:
            try:
                _ = obj.properties.array_spec.pyshape
                base_arr_cls = ftype_explicit_array
            except Exception:
                # Bounds depend on runtime arguments (e.g. 2*n, 2**n).
                base_arr_cls = ftype_assumed_size_array
        elif is_assumed_rank:
            base_arr_cls = ftype_assumed_rank
        elif is_assumed_shape:
            base_arr_cls = ftype_assumed_shape
        else:
            base_arr_cls = ftype_assumed_size_array

        # Create a concrete subclass with _base implemented as an instance method
        def _base(self):
            base_cls = find_ftype(ftype, kind)
            base = base_cls.__new__(base_cls)
            base._symbol = getattr(self, "_symbol", None)
            base._module_obj = getattr(self, "_module_obj", None)
            type(base).__init__(base)  # type: ignore[misc]
            return base

        res = type(base_arr_cls.__name__, (base_arr_cls,), {"_base": _base})
    else:
        res = find_ftype(ftype, kind)

    return res


def find_ftype(ftype: str, kind: int) -> type[f_type]:
    # character is matched by ftype alone
    key: tuple[str, int | None] = (
        ("character", None) if ftype == "character" else (ftype, kind)
    )
    try:
        return _ftype_registry[key]
    except KeyError:
        raise TypeError(f"No f_type registered for ftype={ftype!r}, kind={kind!r}")


class ftype_strlen(ftype_integer):
    @property
    def ctype(self):
        if is_64bit():
            return ctypes.c_int64
        else:
            return ctypes.c_int32


class ftype_optional(f_type):
    ctype = ctypes.c_byte
