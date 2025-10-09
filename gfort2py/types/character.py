# SPDX-License-Identifier: GPL-2.0+i

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Type

import gfModParser as gf
from .base import f_type


class ftype_char(f_type, metaclass=ABCMeta):
    default = ""
    ftype = "character"

    def __init__(self, value=None):
        self._value = value

        super().__init__()

    @property
    @abstractmethod
    def _base_ctype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def encoding(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.ftype}(kind={self.kind},len={self.strlen})"

    @property
    def ctype(self):
        if self.strlen is None:
            return self._base_ctype
        else:
            return self._base_ctype * self.strlen

    @property
    def value(self):
        try:
            self._value = self._ctype.value.decode(self.encoding)
        except AttributeError:
            self._value = str(
                self._ctype.value
            )  # Functions returning str's give us str not bytes
        self.strlen = len(self._value)
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            return None

        if hasattr(value, "encode"):
            value = value.encode()

        if self.strlen is not None:
            if len(value) > self.strlen:
                value = value[: self.strlen]
            else:
                value = value + b" " * (self.strlen - len(value))

        self._value = value
        self._ctype.value = value

    @property
    def strlen(self) -> int | None:
        return self._strlen

    @strlen.setter
    def strlen(self, value) -> int | None:
        self._strlen = value
        return self._strlen


class ftype_character_1(ftype_char):
    kind = 1
    _base_ctype = ctypes.c_char_p
    dtype = np.dtype(np.bytes_)
    encoding = "ascii"


class ftype_character_4(ftype_char):
    kind = 4
    _base_ctype = ctypes.c_wchar_p
    dtype = np.dtype(np.str_)
    encoding = "utf_32"


def init_char(obj: Type[gf.Symbol]) -> ftype_char:
    if obj.kind == 1:
        c = ftype_character_1
    else:
        c = ftype_character_4

    class f_character(c):
        @property
        def strlen(self):
            return obj.properties.parameter.len

    return f_character
