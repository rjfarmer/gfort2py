# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta, abstractmethod
from typing import Any, Type, cast

import gfModParser as gf

from ...types import factory


class fArg(metaclass=ABCMeta):
    def __init__(self, definition: gf.Symbol, procedure: gf.Symbol, module: gf.Module):
        self.definition: gf.Symbol = definition
        self.module: gf.Module = module
        self.procedure: gf.Symbol = procedure

        cls = factory(self.definition)
        c = cls.__new__(cls)
        c._symbol = self.definition
        type(c).__init__(c)  # type: ignore[misc]
        self.base = c
        self._ctype = None

    @property
    def is_optional(self) -> bool:
        return self.definition.properties.attributes.optional

    @property
    def is_intent_out(self) -> bool:
        return self.definition.properties.attributes.intent == "OUT"

    @property
    def is_pointer(self) -> bool:
        return self.definition.properties.attributes.pointer

    @property
    def is_allocatable(self) -> bool:
        return self.definition.properties.attributes.allocatable

    def can_be_unset(self) -> bool:
        return self.is_intent_out or self.is_allocatable or self.is_optional

    @property
    def is_value(self) -> bool:
        return self.definition.properties.attributes.value

    @property
    def is_return(self) -> bool:
        return False

    @property
    def is_character(self) -> bool:
        return self.definition.type.lower() == "character"

    @property
    def needs_resolving_late(self) -> bool:
        return False

    @property
    def ctype(self):
        return self._ctype

    @ctype.setter
    def ctype(self, value):
        if value is None and self.is_optional:
            if self.is_character:
                self._ctype = ctypes.c_void_p(0)
            else:
                self._ctype = None
            return

        self.base.value = value
        if self.is_value:
            self._ctype = self.base._ctype
        elif self.is_pointer:
            self._ctype = self.base.pointer2()
        else:
            self._ctype = self.base.pointer()

    def value(self):
        if self._ctype is None:
            return None

        if (
            self.is_optional
            and self.is_character
            and isinstance(self._ctype, ctypes.c_void_p)
            and self._ctype.value is None
        ):
            return None

        if self.is_value:
            c = self._ctype
        elif self.is_pointer:
            p = cast(Any, self._ctype)
            c = p.contents.contents
        else:
            p = cast(Any, self._ctype)
            c = p.contents

        return self.base.from_ctype(c, symbol=self.definition).value
