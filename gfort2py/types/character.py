# SPDX-License-Identifier: GPL-2.0+i

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod

from ..utils import is_64bit
from .base import f_type
from .integer import f_integer


class f_char(f_type, metaclass=ABCMeta):
    default = ""

    def __init__(self, value=None, strlen=None):
        self._value = value

        if strlen is None and value is not None:
            self.strlen = len(self.value)

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
            value = value.encode(self.encoding)

        if self.strlen is not None:
            if len(value) > self.strlen:
                value = value[: self.strlen]
            else:
                value = value + b" " * (self.strlen - len(value))

        self._value = value
        self._ctype.value = value


class f_character_1(f_char):
    kind = 1
    _base_ctype = ctypes.c_char_p
    dtype = np.dtype(np.bytes_)
    encoding = "ascii"


class f_character_4(f_char):
    kind = 4
    _base_ctype = ctypes.c_wchar_p
    dtype = np.dtype(np.str_)
    encoding = "utf_32"


class f_strstrlen(f_integer):
    @property
    def ctype(self):
        if is_64bit():
            return ctypes.c_int64
        else:
            return ctypes.c_int32
