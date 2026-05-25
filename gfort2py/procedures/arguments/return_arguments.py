# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf

from ...types import factory


class fReturnArguments:
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        self.procedure = procedure
        self.module = module
        self.values = values
        self.return_symbol = return_symbol
        self._ctypes: list[Any] = []

    def set_values(self):
        self._ctypes = []

    def get_ctypes(self) -> list[Any]:
        return self._ctypes

    def get_values(self) -> list[Any]:
        return []


class fReturnCharArguments(fReturnArguments):
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        super().__init__(procedure, module, values, return_symbol)
        self._buffer = None
        self._result_type = None

    def _strlen_ctype(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            return ctypes.c_int64
        return ctypes.c_int32

    def _build_return_type(self):
        cls = factory(self.return_symbol)
        c = cls.__new__(cls)
        c._symbol = self.return_symbol
        type(c).__init__(c)  # type: ignore[misc]
        return c

    def set_values(self):
        self._ctypes = []
        self._result_type = self._build_return_type()

        length = self.return_symbol.properties.typespec.charlen.value
        if length <= 0:
            length = 1

        self._result_type.value = " " * length
        self._buffer = self._result_type._ctype

        self._ctypes.append(self._buffer)
        self._ctypes.append(self._strlen_ctype()(length))

    def get_values(self) -> list[Any]:
        if self._result_type is None or self._buffer is None:
            return []

        value = self._result_type.from_ctype(
            self._buffer, symbol=self.return_symbol
        ).value
        return [value]


class fReturnArrayArguments(fReturnArguments):
    pass


class fReturnDTArguments(fReturnArguments):
    pass
