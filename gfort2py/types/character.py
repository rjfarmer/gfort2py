# SPDX-License-Identifier: GPL-2.0+i

import ctypes
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Type

import gfModParser as gf
from .base import f_type


__all__ = ["ftype_character"]


class ftype_character(f_type, metaclass=ABCMeta):
    default = ""
    ftype = "character"

    def __init__(self, value=None):
        self._value = value

        if self.definition().kind == 1:
            self._char = ftype_character_1(self)
        else:
            self._char = ftype_character_4(self)

        if self.definition().properties.typespec.charlen.value > 0:
            self._length = ftype_char_fixed(self)
        else:
            self._length = ftype_char_defered(self)

        super().__init__()

    def __repr__(self):
        return self._length.__repr__()

    @property
    def ctype(self):
        return self._length.ctype

    @property
    def _base_ctype(self):
        return self._char._base_ctype

    def kind(self):
        return self._char.kind

    @property
    def dtype(self):
        return self._char.dtype

    @property
    def value(self) -> str | None:
        self._char.decode_value(self._ctype)

    @value.setter
    def value(self, value: str | bytes | None):
        if value is None:
            return None

        if hasattr(value, "encode"):
            value = value.encode(self._char.encoding)

        self._length.set_value(value)

    @property
    def strlen(self):
        return self._length.strlen


#####################


class ftype_char(metaclass=ABCMeta):
    default = ""
    ftype = "character"

    def __init__(self, parent, value=None):
        self._value = value
        self.parent = parent

    @property
    @abstractmethod
    def _base_ctype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def encoding(self):
        raise NotImplementedError

    @abstractmethod
    def decode_value(self, ctype):
        raise NotImplementedError


class ftype_character_1(ftype_char):
    kind = 1
    _base_ctype = ctypes.c_char
    dtype = np.dtype(np.bytes_)
    encoding = "ascii"

    def decode_value(self, ctype) -> str:
        try:
            return ctype.value.decode(self.encoding)
        except AttributeError:
            return str(ctype)  # Functions returning str's give us str not bytes


class ftype_character_4(ftype_char):
    kind = 4
    _base_ctype = ctypes.c_char * 4
    dtype = np.dtype(np.str_)
    encoding = "utf16"

    def decode_value(self, ctype) -> bytes:
        return b"".join([ctype[i].value for i in range(self.parent.strlen)]).decode()


##############################


class ftype_char_length(metaclass=ABCMeta):
    def __init__(self, parent):
        self.parent = parent

    @property
    @abstractmethod
    def strlen(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def ctype(self):
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value):
        raise NotImplementedError


class ftype_char_fixed(ftype_char_length):

    @property
    def strlen(self):
        return self.parent.definition().properties.typespec.charlen.value

    def __repr__(self):
        return f"character(kind={self.parent.kind},len={self.strlen})"

    @property
    def ctype(self):
        return self.parent._base_ctype * self.strlen

    def set_value(self, value):
        if len(value) > self.strlen:
            value = value[: self.strlen]
        else:
            value = value + b" " * (self.strlen - len(value))

        self.parent._value = value

        self.parent._ctype.value = value


class ftype_char_defered(ftype_char_length):

    @property
    def strlen(self):
        if self.parent._value is None:
            return -1
        else:
            return len(self.parent._value)

    def __repr__(self):
        l = self.strlen
        if l == -1:
            l = "*"

        return f"character(kind={self.parent.kind},len={l})"

    @property
    def ctype(self):
        if self.parent._value is not None:
            return self.parent._base_ctype * len(self.parent._value)
        else:
            return self.parent._base_ctype

    def set_value(self, value):

        self.parent._value = value

        self.parent._ctype = self.ctype()

        self.parent._ctype.value = value
