# SPDX-License-Identifier: GPL-2.0+

from typing import Type, Any
import ctypes
from abc import ABC, abstractmethod

import gfModParser as gf

from ...types import factory


class fArg(ABC):
    def __init__(
        self, definition: gf.Symbol, procedure: gf.Symbol, module: Type[gf.Module]
    ):
        self.definition: gf.Symbol = definition
        self.module: Type[gf.Module] = module
        self.procedure: gf.Symbol = procedure

        self.base = factory(self.definition)()
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
    def needs_resolving_late(self) -> bool:
        return False

    @property
    def ctype(self):
        return self._ctype

    @ctype.setter
    def ctype(self, value):
        self.base.value = value
        if self.is_value:
            self._ctype = self.base.ctype(value)
        elif self.is_pointer:
            self._ctype = self.base.pointer2()
        else:
            self._ctype = self.base.pointer()

    def value(self):
        if self._ctype is None:
            return None

        if self.is_value:
            c = self._ctype
        elif self.is_pointer:
            c = self._ctype.contents.contents
        else:
            c = self._ctype.contents

        return self.base.from_ctype(c).value
