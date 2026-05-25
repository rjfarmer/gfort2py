# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf

from .arguments import fArguments


class fArgumentsExtra(fArguments):
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        arguments: fArguments,
    ):
        super().__init__(procedure=procedure, module=module, values=values)
        self._ctypes: list[Any] = []
        self._arguments = arguments

    def _strlen_ctype(self):
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            return ctypes.c_int64
        return ctypes.c_int32

    def _string_length(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, bytes):
            return len(value)
        if isinstance(value, str):
            return len(value)
        try:
            return len(value)
        except TypeError:
            return 0

    def set_values(self):
        """Resolve hidden trailing arguments expected by gfortran ABI."""
        self._ctypes = []

        for key in self.procedure.properties.formal_argument:
            symbol = self.module[key]
            name = symbol.name
            arg = self._arguments.args[name]
            is_character = symbol.type.lower() == "character"
            is_allocatable_character = (
                is_character and symbol.properties.attributes.allocatable
            )

            if is_character and not is_allocatable_character:
                length = self._string_length(arg.value)
                self._ctypes.append(self._strlen_ctype()(length))

            if symbol.properties.attributes.optional and not is_character:
                marker = 1 if arg.set and arg.value is not None else 0
                self._ctypes.append(ctypes.c_byte(marker))

    def get_ctypes(self) -> list[Any]:
        return self._ctypes

    def get_values(self) -> dict[str, Any]:
        return {}
