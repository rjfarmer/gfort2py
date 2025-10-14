# SPDX-License-Identifier: GPL-2.0+

from typing import Type

import gfModParser as gf

import ctypes
from abc import ABC, abstractmethod


__all__ = ["f_type"]


class f_type(ABC):
    """
    Base class for interfacing with Fortran objects

    Most class will want to override value getter/setter
    and may need to provide a ctypes() property if it can't be easily
    expressed as a class variable

    Other class variables/properties to be set are:
    ftype: string Fortran type
    kind: int Fortran kind (byte size)
    ctype: python ctype for single object
    base_ctype: If set, the ctype of a single element (use full mostly for complex numbers)
    dtype: numpy dtype
    default: default value for setting element of allocated arrays
    encoding: character types need to store their character encoding

    """

    def __init__(self, value=None):
        self._value = value
        self._ctype = self.ctype()
        self._p1 = None
        self._p2 = None

    @property
    @abstractmethod
    def ftype(self) -> str:
        """Store the Fortran type

        Returns:
            str:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def kind(self) -> int:
        """Store the Fortran kind, assumed to be the bytesize for most objects

        Returns:
            int
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ctype(self):
        """Stores the python ctype class (not instance)

        Returns:
            ctypes.ctype
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self):
        """Stores the numpy dtype

        Returns:
            np.dtype
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def object(self) -> Type[gf.Symbol]:
        """Stores the module data

        Should be injected into the class before creation

        Returns:
            Module data
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.ftype}(kind={self.kind})"

    @property
    def value(self):
        """Gets the value currently stored in the type and convert to python type

        Returns:
            value
        """
        self._value = self._ctype.value
        return self._value

    @value.setter
    def value(self, value):
        """Convert python value to ctype value and save in the ctype

        Args:
            value (Any): Python value compatible with the base ctype
        """
        self._value = value
        self._ctype.value = self._value

    @property
    def _as_parameter_(self):
        """Returns the ctype instance"""
        return self._ctype

    @classmethod
    def in_dll(cls, lib, name):
        c = cls()
        c._ctype = c.ctype.in_dll(lib, name)
        return c

    @classmethod
    def from_param(cls, obj):
        c = cls(obj)
        return c.ctype

    @classmethod
    def from_address(cls, address):
        c = cls()
        c._ctype = c.ctype.from_address(address)
        return c

    def byref(self):
        return ctypes.byref(self._ctype)

    def pointer(self):
        self._p1 = ctypes.pointer(self._ctype)
        return self._p1

    def pointer2(self):
        if self._p1 is None:
            _ = self.pointer()
        self._p2 = ctypes.pointer(self._p1)
        return self._p2
