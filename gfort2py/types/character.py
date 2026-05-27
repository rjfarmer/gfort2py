# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Type

import gfModParser as gf
import numpy as np

from ..compilation import Modulise
from ..utils import get_c_runtime, strlen_ctype
from .base import f_type

__all__ = ["ftype_character"]


class ftype_character(f_type, metaclass=ABCMeta):
    default = "''"
    ftype = "character"

    def __init__(self, value=None):
        self._alloc_len_ctype = None
        self._value = None
        super().__init__(value=value)

    @staticmethod
    def _charlen_as_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def in_dll(
        cls, lib: ctypes.CDLL, name: str, *, symbol: gf.Symbol | None = None
    ) -> "ftype_character":
        c = cls.__new__(cls)
        c._symbol = symbol
        c.__init__()  # type: ignore[misc]

        declared_len = None
        if symbol is not None:
            declared_len = cls._charlen_as_int(symbol.properties.typespec.charlen.value)

        is_alloc_deferred = (
            symbol is not None
            and symbol.properties.attributes.allocatable
            and (declared_len is None or declared_len <= 0)
        )

        if is_alloc_deferred:
            c._ctype = ctypes.c_void_p.in_dll(lib, name)
            c._alloc_len_ctype = None
            if name.startswith("__") and "_MOD_" in name:
                module_and_rest = name[2:]
                len_symbol = f"_F.{module_and_rest}"
                try:
                    c._alloc_len_ctype = strlen_ctype().in_dll(lib, len_symbol)
                except ValueError:
                    c._alloc_len_ctype = None
            return c

        c._ctype = c.ctype.in_dll(lib, name)
        return c

    @cached_property
    def _char(self):
        if self._sym.kind == 1:
            return ftype_character_1(self)
        return ftype_character_4(self)

    @cached_property
    def _length(self):
        declared_len = self._charlen_as_int(self._sym.properties.typespec.charlen.value)
        if declared_len is not None and declared_len > 0:
            return ftype_char_fixed(self)
        return ftype_char_defered(self)

    def __repr__(self):
        return self._length.__repr__()

    @property
    def ctype(self):
        return self._length.ctype

    @property
    def _base_ctype(self):
        return self._char._base_ctype

    @property
    def kind(self):
        return self._char.kind

    @property
    def dtype(self):
        return self._char.dtype

    @property
    def value(self) -> str | None:
        declared_len = self._charlen_as_int(self._sym.properties.typespec.charlen.value)
        is_alloc_deferred = (
            self._sym.properties.attributes.allocatable
            and (declared_len is None or declared_len <= 0)
            and isinstance(self._ctype, ctypes.c_void_p)
        )

        if is_alloc_deferred:
            ptr = self._ctype.value
            if ptr is None:
                return None

            if self._alloc_len_ctype is not None:
                strlen = int(self._alloc_len_ctype.value)
            else:
                strlen = len(ctypes.string_at(ptr))

            if strlen <= 0:
                return ""

            return ctypes.string_at(ptr, strlen).decode(self._char.encoding)

        return self._char.decode_value(self._ctype)

    @value.setter
    def value(self, value: str | bytes | None):
        declared_len = self._charlen_as_int(self._sym.properties.typespec.charlen.value)
        is_alloc_deferred = (
            self._sym.properties.attributes.allocatable
            and (declared_len is None or declared_len <= 0)
            and isinstance(self._ctype, ctypes.c_void_p)
        )

        if is_alloc_deferred:
            libc = get_c_runtime()
            libc.malloc.argtypes = [ctypes.c_size_t]
            libc.malloc.restype = ctypes.c_void_p
            libc.realloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.realloc.restype = ctypes.c_void_p

            if value is None:
                self._ctype.value = None
                if self._alloc_len_ctype is not None:
                    self._alloc_len_ctype.value = 0
                return None

            if hasattr(value, "encode"):
                value = value.encode(self._char.encoding)

            strlen = len(value)
            current_ptr = self._ctype.value

            if current_ptr is None:
                new_ptr = libc.malloc(strlen)
            else:
                new_ptr = libc.realloc(current_ptr, strlen)

            if not new_ptr and strlen > 0:
                raise MemoryError("Unable to allocate memory for allocatable character")

            if strlen > 0:
                ctypes.memmove(new_ptr, value, strlen)

            self._ctype.value = new_ptr
            if self._alloc_len_ctype is not None:
                self._alloc_len_ctype.value = strlen
            return None

        if value is None:
            return None

        if hasattr(value, "encode"):
            value = value.encode(self._char.encoding)

        self._length.set_value(value)

    @property
    def strlen(self):
        return self._length.strlen

    def allocate(self, shape) -> Modulise:
        dims = ",".join([":"] * len(shape))

        shape = ",".join([str(i) for i in shape])

        string = f"""
        subroutine alloc(x)
        {self.ftype}(kind={self.kind},len={self.strlen}),allocatable,dimension({dims}), intent(out) :: x
        if(allocated(x)) deallocate(x)
        allocate(x({shape}))
        x = {self.default}
        end subroutine alloc
        """
        return Modulise(string)


#####################


class ftype_char(metaclass=ABCMeta):
    default = ""
    ftype = "character"

    def __init__(self, parent: ftype_character, value=None):
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
    encoding = "utf-8"

    def decode_value(self, ctype) -> str:
        raw = bytes([ctype[i].value[0] for i in range(self.parent.strlen)])
        return raw.decode(self.encoding)


##############################


class ftype_char_length(metaclass=ABCMeta):
    def __init__(self, parent: ftype_character):
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
        length = ftype_character._charlen_as_int(
            self.parent._sym.properties.typespec.charlen.value
        )
        if length is None:
            raise ValueError("Character length is not a fixed integer")
        return length

    def __repr__(self):
        return f"character(kind={self.parent.kind},len={self.strlen})"

    @property
    def ctype(self):
        return self.parent._char._base_ctype * self.strlen

    def set_value(self, value):
        if self.parent.kind == 4:
            if hasattr(value, "encode"):
                value = value.encode(self.parent._char.encoding)

            capacity = self.strlen
            pad = " ".encode(self.parent._char.encoding)
            if len(value) > capacity:
                value = value[:capacity]
            else:
                value = value + pad * (capacity - len(value))

            raw = b"".join(bytes([byte, 0, 0, 0]) for byte in value)
            self.parent._char._value = value
            ctypes.memmove(ctypes.addressof(self.parent._ctype), raw, len(raw))
            return

        if len(value) > self.strlen:
            value = value[: self.strlen]
        else:
            value = value + b" " * (self.strlen - len(value))

        self.parent._char._value = value

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
        if self.parent._char._value is not None:
            return self.parent._char._base_ctype * len(self.parent._char._value)
        else:
            return self.parent._char._base_ctype

    def set_value(self, value):
        if self.parent.kind == 4:
            if hasattr(value, "encode"):
                value = value.encode(self.parent._char.encoding)

            self.parent._char._value = value
            self.parent._ctype = self.ctype()
            raw = b"".join(bytes([byte, 0, 0, 0]) for byte in value)
            ctypes.memmove(ctypes.addressof(self.parent._ctype), raw, len(raw))
            return

        self.parent._char._value = value

        self.parent._ctype = self.ctype()

        self.parent._ctype.value = value
