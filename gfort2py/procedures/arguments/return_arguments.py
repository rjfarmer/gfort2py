# SPDX-License-Identifier: GPL-2.0+

import ctypes
import sys
from typing import Any

import gfModParser as gf
import numpy as np

from .platform import is_windows

try:
    import pyquadp as pyq  # type: ignore[import-not-found]

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

from ...compilation import Compile, Modulise
from ...types import factory, find_ftype
from ...types.arrays import ftype_assumed_shape
from ...types.dt import ftype_dt_assumed_shape
from ...utils import strlen_ctype

_ALLOC_CHAR_DEALLOCATOR: dict[int, tuple[ctypes.CDLL, Any]] = {}


def _alloc_char_deallocator(kind: int) -> Any:
    if kind in _ALLOC_CHAR_DEALLOCATOR:
        _lib, sub = _ALLOC_CHAR_DEALLOCATOR[kind]
        return sub

    code = Modulise(f"""
        subroutine dealloc_char(x)
            character(kind={kind},len=:), allocatable, intent(inout) :: x
            if (allocated(x)) deallocate(x)
        end subroutine dealloc_char
        """)
    comp = Compile(code.as_module(), name=code.strhash())
    if not comp.compile():
        raise RuntimeError("Failed to compile deferred character deallocator")

    lib = comp.platform.load_library(comp.library_filename)
    sub = getattr(lib, f"__{comp.name}_MOD_dealloc_char")
    _ALLOC_CHAR_DEALLOCATOR[kind] = (lib, sub)
    return sub


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

    def _apply_interface_op(self, op_name: str, left: Any, right: Any) -> Any:
        if op_name == "PARENTHESES":
            return left

        op = self.module.interface.op(op_name)
        if op is None:
            raise NotImplementedError(f"Operator {op_name} is not supported")

        return op(left, right)

    def _resolve_function_expression(self, exp_type: Any, ctx: dict[str, Any]) -> int:
        raw = getattr(exp_type, "_args", None)
        if not isinstance(raw, list) or len(raw) < 5:
            raise ValueError("Function expression payload is malformed")

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
            expr_cls = type(exp_type)
            version = getattr(exp_type, "version", None)

            for item in actual_args:
                if len(item) < 2 or not item[1]:
                    continue

                arg_raw = item[1]
                if not isinstance(arg_raw, list) or not arg_raw:
                    continue

                if arg_raw[0] == "VARIABLE":
                    resolved_args.append(
                        self._lookup_context_value(str(arg_raw[3]), ctx)
                    )
                elif arg_raw[0] == "CONSTANT":
                    resolved_args.append(int(str(arg_raw[3]).strip("'\"")))
                else:
                    arg_expr = expr_cls(arg_raw, version=version)
                    resolved_args.append(self._resolve_expression(arg_expr, ctx))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Function expression argument payload is malformed"
            ) from exc

        if func_ref.isdigit() and self._lib is not None:
            ref = int(func_ref)
            try:
                sym = self.module[ref]
            except KeyError:
                sym = None

            if (
                sym is not None
                and sym.is_procedure
                and hasattr(self._lib, sym.mangled_name)
            ):
                from ...procedures import factory as proc_factory

                proc = proc_factory(sym)(self._lib, sym, self.module)
                return int(proc(*resolved_args).result)

        if func_name == "size":
            if not resolved_args:
                raise ValueError("SIZE requires at least one argument")
            return int(np.size(resolved_args[0]))

        raise NotImplementedError(f"Unsupported function {func_name!r}")

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
            return self._resolve_function_expression(exp_type, ctx)

        value = getattr(expr, "value", expr)

        if isinstance(value, (int, np.integer, bool, np.bool_)):
            return int(value)

        if isinstance(value, str):
            try:
                return int(self._lookup_context_value(value, ctx))
            except KeyError:
                pass
            raise ValueError(f"Expression variable {value!r} is unresolved at runtime")

        if value is None:
            raise ValueError("Expression is unresolved at runtime")

        return int(value)


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

    def _resolve_runtime_strlen(self) -> int:
        expr = self.return_symbol.properties.typespec.charlen
        ctx = self._arg_context()
        return self._resolve_expression(expr, ctx)

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


class fReturnAllocCharArguments(fReturnArguments):
    def __init__(
        self,
        procedure: gf.Symbol,
        module: gf.Module,
        lib: ctypes.CDLL,
        values: tuple[tuple[Any, ...], dict[str, Any]],
        return_symbol: gf.Symbol,
    ):
        super().__init__(procedure, module, lib, values, return_symbol)
        self._result_type: Any = None
        self._result_data = ctypes.c_void_p(None)
        self._result_len = strlen_ctype()(0)

    def _build_return_type(self):
        cls = factory(self.return_symbol)
        c = cls.__new__(cls)
        c._symbol = self.return_symbol
        type(c).__init__(c)  # type: ignore[misc]
        return c

    def set_values(self):
        self._ctypes = []
        self._result_type = self._build_return_type()
        self._result_data = ctypes.c_void_p(None)
        self._result_len = strlen_ctype()(0)
        self._ctypes.append(ctypes.pointer(self._result_data))
        self._ctypes.append(ctypes.pointer(self._result_len))

    def get_values(self) -> list[Any]:
        ptr = self._result_data.value
        if ptr is None:
            return [None]

        strlen = int(self._result_len.value)
        if strlen <= 0:
            return [""]

        kind = int(self.return_symbol.kind)
        if kind == 4:
            raw = ctypes.string_at(ptr, strlen * 4)
            units = [
                int.from_bytes(raw[i * 4 : (i + 1) * 4], byteorder=sys.byteorder)
                for i in range(strlen)
            ]

            # Some runtimes lower kind=4 deferred strings as UTF-8 bytes packed
            # into 4-byte elements (low byte carries data). Detect and decode.
            if all(0 <= unit <= 0xFF for unit in units):
                lane_bytes = bytes(unit for unit in units if unit != 0)
                try:
                    return [lane_bytes.decode("utf-8")]
                except UnicodeDecodeError:
                    return [lane_bytes.decode("latin-1")]

            valid_scalar = all(
                0 <= unit <= 0x10FFFF and not (0xD800 <= unit <= 0xDFFF)
                for unit in units
            )
            if valid_scalar:
                return ["".join(chr(unit) for unit in units if unit != 0)]

            # Last-resort: some compilers report byte length even for kind=4.
            try:
                return [ctypes.string_at(ptr, strlen).decode("utf-8")]
            except UnicodeDecodeError:
                return [ctypes.string_at(ptr, strlen).decode("latin-1")]

        encoding = self._result_type._char.encoding
        data = ctypes.string_at(ptr, strlen)
        try:
            return [data.decode(encoding)]
        except UnicodeDecodeError:
            return [data.decode("utf-8")]

    def release(self) -> None:
        ptr = self._result_data.value
        if ptr is None:
            return

        if is_windows():
            # Windows deallocate is flaky and sometimes crashes, so leak instead of freeing memory.
            return

        dealloc = _alloc_char_deallocator(int(self.return_symbol.kind))
        dealloc(ctypes.pointer(self._result_data), ctypes.pointer(self._result_len))
        self._result_data.value = None
        self._result_len.value = 0


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

        if self.return_symbol.properties.array_spec.is_explicit:
            shape = self._resolve_shape()
            if self.return_symbol.is_dt:
                self._result_type._ensure_shape(shape)
            else:
                dtype = self._result_type.base.dtype
                initial = np.zeros(shape, dtype=dtype, order="F")
                self._result_type.value = initial

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
