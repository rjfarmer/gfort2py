# SPDX-License-Identifier: GPL-2.0+

import ctypes
from typing import Any

import gfModParser as gf
import numpy as np

from ...types import factory, find_ftype
from ...types.arrays import ftype_assumed_shape
from ...types.dt import ftype_dt_assumed_shape
from ...utils import strlen_ctype


class fReturnArguments:
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        lib: ctypes.CDLL,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        self.procedure = procedure
        self.module = module
        self._lib = lib
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
        lib: ctypes.CDLL,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        super().__init__(procedure, module, lib, values, return_symbol)
        self._buffer = None
        self._result_type = None
        self._runtime_strlen: int | None = None

    def _build_return_type(self):
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

    def _lookup_context_value(self, key: str, ctx: dict[str, Any]) -> Any:
        if key in ctx:
            return ctx[key]

        if key.isdigit():
            symbol = self.module[int(key)]
            if symbol.name in ctx:
                return ctx[symbol.name]

        raise KeyError(key)

    def _resolve_runtime_strlen(self) -> int:
        expr = self.return_symbol.properties.typespec.charlen
        value = expr.value
        ctx = self._arg_context()

        exp_type = getattr(expr, "type", None)
        if exp_type is not None and type(exp_type).__name__ == "ExpFunction":
            raw = getattr(exp_type, "_args", None)
            if not isinstance(raw, list) or len(raw) < 5:
                raise ValueError("Character length FUNCTION payload is malformed")

            func_ref = str(raw[3]) if len(raw) > 3 else ""
            func_name = ""
            for idx in (7, 5):
                if idx < len(raw):
                    candidate = str(raw[idx]).strip("'\"").lower()
                    if candidate and not candidate.isdigit():
                        func_name = candidate
                        break

            try:
                actual_args = raw[4]
                resolved_args: list[Any] = []
                for item in actual_args:
                    if len(item) < 2 or not item[1]:
                        continue

                    arg_raw = item[1]
                    if isinstance(arg_raw, list) and len(arg_raw):
                        if arg_raw[0] == "VARIABLE":
                            resolved_args.append(
                                self._lookup_context_value(str(arg_raw[3]), ctx)
                            )
                        elif arg_raw[0] == "CONSTANT":
                            resolved_args.append(int(str(arg_raw[3]).strip("'\"")))
                        else:
                            arg_expr = type(expr)(arg_raw, version=expr.version)
                            arg_val = arg_expr.value
                            if isinstance(arg_val, str):
                                arg_val = self._lookup_context_value(arg_val, ctx)
                            resolved_args.append(arg_val)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Character length FUNCTION argument payload is malformed"
                ) from exc

            if func_ref.isdigit():
                ref = int(func_ref)
                try:
                    sym = self.module[ref]
                except KeyError:
                    sym = None

                if sym is not None and sym.is_procedure and self._lib:
                    from ...procedures import factory as proc_factory

                    proc = proc_factory(sym)(self._lib, sym, self.module)
                    return int(proc(*resolved_args).result)

            if func_name == "size":
                if not resolved_args:
                    raise ValueError("SIZE requires at least one argument")
                return int(np.size(resolved_args[0]))

            raise NotImplementedError(
                f"Unsupported runtime character length function {func_name!r}"
            )

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, str):
            try:
                return int(self._lookup_context_value(value, ctx))
            except KeyError:
                pass

        raise ValueError("Character length is unresolved at runtime")

    def set_values(self):
        self._ctypes = []
        self._result_type = self._build_return_type()

        length = self._resolve_runtime_strlen()
        if length <= 0:
            length = 1
        self._runtime_strlen = length

        self._result_type.value = " " * length
        self._buffer = self._result_type._ctype

        self._ctypes.append(self._buffer)
        self._ctypes.append(strlen_ctype()(length))

    def get_values(self) -> list[Any]:
        if self._result_type is None or self._buffer is None:
            return []

        value = self._result_type.from_ctype(
            self._buffer, symbol=self.return_symbol
        ).value

        if isinstance(value, str) and self._runtime_strlen is not None:
            value = value[: self._runtime_strlen]

        return [value]


class fReturnArrayArguments(fReturnArguments):
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        lib: ctypes.CDLL,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        super().__init__(procedure, module, lib, values, return_symbol)
        self._buffer = None
        self._result_type = None

    def _build_return_type(self):
        if self.return_symbol.properties.array_spec.is_explicit:
            if self.return_symbol.is_dt:
                # Explicit-shape DT function results are returned through a
                # descriptor hidden argument in this ABI; use descriptor-backed
                # storage so dims/offset/data are initialized before the call.
                cls = ftype_dt_assumed_shape
            else:
                ftype = self.return_symbol.type.lower()
                kind = self.return_symbol.kind

                def _base(self):
                    base_cls = find_ftype(ftype, kind)
                    base = base_cls.__new__(base_cls)
                    base._symbol = getattr(self, "_symbol", None)
                    base._module_obj = getattr(self, "_module_obj", None)
                    type(base).__init__(base)  # type: ignore[misc]
                    return base

                cls = type(
                    "ftype_return_always_explicit",
                    (ftype_assumed_shape,),
                    {"_base": _base},
                )
        else:
            cls = factory(self.return_symbol)

        c = cls.__new__(cls)
        c._symbol = self.return_symbol
        if self.return_symbol.is_dt:
            c._module_obj = self.module
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

    def _apply_interface_op(self, op_name: str, left: Any, right: Any) -> Any:
        if op_name == "PARENTHESES":
            return left

        op = self.module.interface.op(op_name)
        if op is None:
            raise NotImplementedError(f"Operator {op_name} is not supported")

        return op(left, right)

    def _lookup_context_value(self, key: str, ctx: dict[str, Any]) -> Any:
        if key in ctx:
            return ctx[key]

        if key.isdigit():
            symbol = self.module[int(key)]
            if symbol.name in ctx:
                return ctx[symbol.name]

        raise KeyError(key)

    def _resolve_size_function(self, exp_type: Any, ctx: dict[str, Any]) -> int:
        raw = getattr(exp_type, "_args", None)
        if not isinstance(raw, list) or len(raw) < 5:
            raise ValueError("SIZE expression payload is malformed")

        func_ref = str(raw[3]) if len(raw) > 3 else ""
        func_name = ""
        for idx in (7, 5):
            if idx < len(raw):
                candidate = str(raw[idx]).strip("'\"").lower()
                if candidate and not candidate.isdigit():
                    func_name = candidate
                    break

        try:
            actual_args = raw[4]
            resolved_args: list[Any] = []
            for item in actual_args:
                if len(item) < 2 or not item[1]:
                    continue

                arg_raw = item[1]
                if isinstance(arg_raw, list) and len(arg_raw):
                    if arg_raw[0] == "VARIABLE":
                        resolved_args.append(
                            self._lookup_context_value(str(arg_raw[3]), ctx)
                        )
                    elif arg_raw[0] == "CONSTANT":
                        resolved_args.append(int(str(arg_raw[3]).strip("'\"")))
                    else:
                        raise ValueError(
                            "Unsupported function argument expression payload"
                        )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Function expression argument payload is malformed"
            ) from exc

        if func_ref.isdigit():
            ref = int(func_ref)
            try:
                sym = self.module[ref]
            except KeyError:
                sym = None

            if sym is not None and sym.is_procedure and self._lib:
                from ...procedures import factory as proc_factory

                try:
                    proc = proc_factory(sym)(self._lib, sym, self.module)
                    return int(proc(*resolved_args).result)
                except AttributeError:
                    # Intrinsic-like helper symbols may exist in mod metadata but
                    # have no callable symbol in the shared library.
                    pass

        if func_name == "size":
            if not resolved_args:
                raise ValueError("SIZE requires at least one argument")
            return int(np.size(resolved_args[0]))

        raise NotImplementedError(f"Unsupported bound function {func_name!r}")

    def _resolve_expression(self, expr: Any, ctx: dict[str, Any]) -> int:
        exp_type = getattr(expr, "type", None)

        if exp_type is not None and hasattr(exp_type, "unary_op"):
            op_name = exp_type.unary_op
            try:
                left_expr, right_expr = exp_type.unary_args
            except (AttributeError, IndexError, TypeError, ValueError):
                raw = getattr(exp_type, "_args", None)
                if not isinstance(raw, list) or len(raw) < 5:
                    raise ValueError("Operator expression payload is malformed")

                expr_cls = type(expr)
                version = expr.version
                left_raw = raw[4]
                if not left_raw:
                    raise ValueError("Operator expression left operand is missing")

                left_expr = expr_cls(left_raw, version=version)

                if op_name == "PARENTHESES":
                    return self._resolve_expression(left_expr, ctx)

                right_raw = raw[5] if len(raw) > 5 else None
                if not right_raw:
                    raise ValueError("Operator expression right operand is missing")

                right_expr = expr_cls(right_raw, version=version)

            left = self._resolve_expression(left_expr, ctx)
            right = self._resolve_expression(right_expr, ctx)
            return int(self._apply_interface_op(op_name, left, right))

        if exp_type is not None and type(exp_type).__name__ == "ExpFunction":
            return self._resolve_size_function(exp_type, ctx)

        value = getattr(expr, "value", expr)

        if isinstance(value, (int, np.integer, bool, np.bool_)):
            return int(value)

        if isinstance(value, str):
            try:
                return int(self._lookup_context_value(value, ctx))
            except KeyError:
                pass

            raise ValueError(f"Array bound variable {value!r} is unresolved at runtime")

        if value is None:
            raise ValueError("Array bound is unresolved at runtime")

        return int(value)

    def _resolve_bound(self, expr: Any, ctx: dict[str, Any]) -> int:
        return self._resolve_expression(expr, ctx)

    def _resolve_shape(self) -> tuple[int, ...]:
        try:
            return tuple(
                int(i) for i in self.return_symbol.properties.array_spec.pyshape
            )
        except (AttributeError, TypeError, ValueError):
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

        if (
            self.return_symbol.properties.array_spec.is_explicit
            and not self.return_symbol.is_dt
        ):
            shape = self._resolve_shape()
            initial = np.zeros(shape, dtype=self._result_type.base.dtype, order="F")
            self._result_type.value = initial
        elif (
            self.return_symbol.properties.array_spec.is_explicit
            and self.return_symbol.is_dt
        ):
            shape = self._resolve_shape()
            self._result_type._ensure_shape(shape)

        self._buffer = self._result_type.pointer()
        self._ctypes.append(self._buffer)

        if self.return_symbol.type.lower() == "character":
            length = int(self.return_symbol.properties.typespec.charlen.value)
            if length <= 0:
                length = 1
            self._ctypes.append(strlen_ctype()(length))

    def get_values(self) -> list[Any]:
        if self._result_type is None:
            return []

        return [self._result_type.value]


class fReturnDTArguments(fReturnArguments):
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        lib: ctypes.CDLL,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        super().__init__(procedure, module, lib, values, return_symbol)
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
