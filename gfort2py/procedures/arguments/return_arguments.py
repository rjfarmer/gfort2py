# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf
import numpy as np

from ...types import factory, find_ftype
from ...types.arrays import ftype_assumed_shape


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

    def _build_return_type(self):
        if self.return_symbol.properties.attributes.always_explicit:
            ftype = self.return_symbol.type.lower()
            kind = self.return_symbol.kind

            def _base(self):
                return find_ftype(ftype, kind)()

            cls = type(
                "ftype_return_always_explicit",
                (ftype_assumed_shape,),
                {"_base": _base},
            )
        else:
            cls = factory(self.return_symbol)

        c = cls.__new__(cls)
        c._symbol = self.return_symbol
        type(c).__init__(c)  # type: ignore[misc]
        return c

    def _arg_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        refs = self.procedure.properties.formal_argument
        names = [self.module[i].name for i in refs]

        for key, value in zip(names, self.values[0]):
            ctx[key] = value

        for key, value in self.values[1].items():
            ctx[key] = value

        for ref in refs:
            name = self.module[ref].name
            if name in ctx:
                ctx[str(ref)] = ctx[name]

        return ctx

    def _resolve_bound(self, expr: Any, ctx: dict[str, Any]) -> int:
        value = getattr(expr, "value", expr)

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, str):
            if value in ctx:
                return int(ctx[value])
            if value.isdigit():
                return int(value)

        if value is None:
            raise ValueError("Array bound is unresolved at runtime")

        return int(value)

    def _resolve_shape(self) -> tuple[int, ...]:
        try:
            return tuple(
                int(i) for i in self.return_symbol.properties.array_spec.pyshape
            )
        except Exception:
            pass

        ctx = self._arg_context()
        lower = self.return_symbol.properties.array_spec.lower
        upper = self.return_symbol.properties.array_spec.upper

        shape = []
        for lo, up in zip(lower, upper):
            l = self._resolve_bound(lo, ctx)
            u = self._resolve_bound(up, ctx)
            shape.append(u - l + 1)

        return tuple(shape)

    def set_values(self):
        self._ctypes = []
        self._result_type = self._build_return_type()

        if self.return_symbol.properties.attributes.always_explicit:
            shape = self._resolve_shape()
            initial = np.zeros(shape, dtype=self._result_type.base.dtype, order="F")
            self._result_type.value = initial

        self._buffer = self._result_type.pointer()
        self._ctypes.append(self._buffer)

    def get_values(self) -> list[Any]:
        if self._result_type is None:
            return []

        return [self._result_type.value]


class fReturnDTArguments(fReturnArguments):
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

    def _build_return_type(self):
        cls = factory(self.return_symbol)
        c = cls.__new__(cls)
        c._symbol = self.return_symbol
        c._module_obj = self.module
        type(c).__init__(c)  # type: ignore[misc]
        return c

    def set_values(self):
        self._ctypes = []
        self._result_type = self._build_return_type()
        self._buffer = self._result_type.pointer()
        self._ctypes.append(self._buffer)

    def get_values(self) -> list[Any]:
        if self._result_type is None:
            return []

        return [self._result_type.value]
