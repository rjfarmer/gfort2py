# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple, Type

import gfModParser as gf
import numpy as np

from ..compilation import Compile, CompileArgs
from ..utils import copy_array, is_64bit
from .base import AllocStrategy, f_type
from .character import ftype_character


class AllocationError(Exception):
    pass


# Cache compiled allocator entry points by (generated module name, compile args)
# so repeated array marshaling avoids spawning the compiler each call.
_ALLOCATOR_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


class ftype_explicit_array(f_type, metaclass=ABCMeta):
    dtype = None  # type: ignore[assignment]
    ftype = None  # type: ignore[assignment]
    kind = None  # type: ignore[assignment]

    def __init__(self, value=None):
        self.base = self._base()
        super().__init__(value=value)

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def ctype(self):
        return self.base.ctype * self.size

    def __repr__(self):
        s = ",".join([str(i) for i in self.shape])
        return f"{self.base.ftype}(kind={self.base.kind})({s})"

    def _is_character_array(self) -> bool:
        return isinstance(self.base, ftype_character)

    def _character_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        strlen = int(self._sym.properties.typespec.charlen.value)
        if strlen <= 0 and value is not None:
            itemsize = int(np.asarray(value).dtype.itemsize)
            if self.base.kind == 4:
                strlen = max(itemsize // 4, 1)
            else:
                strlen = itemsize
        if strlen <= 0 and hasattr(self, "_ctype") and self._ctype is not None:
            ctype_elem = getattr(type(self._ctype), "_type_", None)
            if ctype_elem is not None:
                size = int(ctypes.sizeof(ctype_elem))
                strlen = max(size // 4, 1) if self.base.kind == 4 else size
        if strlen <= 0:
            strlen = 1
        return np.dtype(f"U{strlen}") if self.base.kind == 4 else np.dtype(f"S{strlen}")

    def _array_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        if self._is_character_array():
            return self._character_dtype(value)
        return self.base.dtype

    @property
    def value(self) -> np.ndarray:
        if self._is_character_array():
            if self.base.kind == 4:
                elem_size = ctypes.sizeof(self._ctype._type_)
                raw = ctypes.string_at(
                    ctypes.addressof(self._ctype), self.size * elem_size
                )
                values = []
                for i in range(self.size):
                    chunk = raw[i * elem_size : (i + 1) * elem_size]
                    utf8_bytes = bytes(chunk[j] for j in range(0, elem_size, 4))
                    values.append(utf8_bytes.decode("utf-8").rstrip())
                self._value = np.array(values, dtype=np.str_).reshape(
                    self.shape, order="F"
                )
                return self._value

            dtype = self._array_dtype()
            strlen = int(dtype.itemsize)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), self.size * strlen)
            self._value = (
                np.frombuffer(raw, dtype=dtype, count=self.size)
                .copy()
                .reshape(self.shape, order="F")
            )
            return self._value

        self._value = (
            np.ctypeslib.as_array(self._ctype)
            .reshape(self.shape, order="F")
            .astype(self._array_dtype())
        )
        return self._value

    @value.setter
    def value(self, value: np.ndarray):
        if value is None:
            return None

        if (
            self._is_character_array()
            and self._sym.properties.typespec.charlen.value <= 0
        ):
            strlen = int(np.asarray(value).dtype.itemsize)
            if strlen <= 0:
                strlen = 1
            arr_type = (ctypes.c_char * strlen) * self.size
            if not isinstance(self._ctype, arr_type):
                self._ctype = arr_type()

        self._value = self._array_check(value)
        elem_size = ctypes.sizeof(self.base.ctype)
        if self._is_character_array():
            if self.base.kind == 4:
                flat = (
                    np.asfortranarray(value)
                    .astype(np.str_, copy=False)
                    .ravel(order="F")
                )
                if self._sym.properties.typespec.charlen.value > 0:
                    utf8_len = int(self._sym.properties.typespec.charlen.value)
                    elem_size = utf8_len * 4
                else:
                    utf8_len = max(len(str(item).encode("utf-8")) for item in flat)
                    elem_size = utf8_len * 4
                    arr_type = (ctypes.c_char * elem_size) * self.size
                    self._ctype = arr_type()

                base_addr = ctypes.addressof(self._ctype)
                for idx, item in enumerate(flat):
                    encoded = item.encode("utf-8")
                    if len(encoded) > utf8_len:
                        encoded = encoded[:utf8_len]
                    else:
                        encoded = encoded + b" " * (utf8_len - len(encoded))

                    raw = b"".join(bytes([byte, 0, 0, 0]) for byte in encoded)
                    ctypes.memmove(base_addr + idx * elem_size, raw, len(raw))
                return

            elem_size = int(self._array_dtype(value).itemsize)
        copy_array(
            self._value.ctypes.data,
            ctypes.addressof(self._ctype),
            elem_size,
            self.size,
        )

    def _array_check(self, value):
        if self._is_character_array():
            if self.base.kind == 1 and not np.issubdtype(value.dtype, np.bytes_):
                raise TypeError("Character strings must be bytes (S dtype)")
            if self.base.kind == 4 and not np.issubdtype(value.dtype, np.str_):
                raise TypeError("Unicode strings must be unicode (U dtype)")

        value = np.asfortranarray(value)
        value = value.astype(self._array_dtype(value), copy=False)

        if value.ndim != self.ndims:
            raise ValueError(
                f"Wrong number of dimensions, got {value.ndim} expected {self.ndims}"
            )

        value = value.ravel(order="F")
        return value

    @property
    def shape(self) -> tuple[int, ...]:
        return self._sym.properties.array_spec.pyshape

    @property
    def ndims(self) -> int:
        return self._sym.properties.array_spec.rank

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


class ftype_assumed_size_array(f_type, metaclass=ABCMeta):
    """Fortran assumed-size array argument: integer, intent(inout) :: x(*)

    The size is unknown from the .mod file; it is determined entirely by the
    numpy array passed in by the caller.  The ABI is a plain pointer to
    contiguous data — identical to an explicit array, but without a fixed size.
    """

    dtype = None  # type: ignore[assignment]
    ftype = None  # type: ignore[assignment]
    kind = None  # type: ignore[assignment]

    def __init__(self, value=None):
        self.base = self._base()
        self._shape: tuple[int, ...] | None = None
        # Cannot call super().__init__() because the ctype size is unknown until
        # a value is provided.
        self._ctype = None
        self._p1 = None
        self._p2 = None
        self.value = value

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def ctype(self):
        if self._ctype is None:
            raise RuntimeError(
                "No value has been set; ctype size is unknown for assumed-size arrays"
            )
        return type(self._ctype)

    def __repr__(self):
        return f"{self.base.ftype}(kind={self.base.kind})(*)"

    def _is_character_array(self) -> bool:
        return isinstance(self.base, ftype_character)

    def _character_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        strlen = int(self._sym.properties.typespec.charlen.value)
        if strlen <= 0 and value is not None:
            itemsize = int(np.asarray(value).dtype.itemsize)
            if self.base.kind == 4:
                strlen = max(itemsize // 4, 1)
            else:
                strlen = itemsize
        if strlen <= 0 and self._ctype is not None:
            ctype_elem = getattr(type(self._ctype), "_type_", None)
            if ctype_elem is not None:
                size = int(ctypes.sizeof(ctype_elem))
                strlen = max(size // 4, 1) if self.base.kind == 4 else size
        if strlen <= 0:
            strlen = 1
        return np.dtype(f"U{strlen}") if self.base.kind == 4 else np.dtype(f"S{strlen}")

    def _array_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        if self._is_character_array():
            return self._character_dtype(value)
        return self.base.dtype

    @property
    def value(self) -> Optional[np.ndarray]:
        if self._ctype is None:
            return None

        if self._is_character_array():
            dtype = self._array_dtype()
            n = len(self._ctype)
            strlen = int(dtype.itemsize)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), n * strlen)
            arr = np.frombuffer(raw, dtype=dtype, count=n).copy()
            if self._shape is not None:
                arr = arr.reshape(self._shape, order="F")
            return arr

        arr = np.ctypeslib.as_array(self._ctype).astype(self._array_dtype())
        if self._shape is not None:
            arr = arr.reshape(self._shape, order="F")

        return arr

    @value.setter
    def value(self, value: np.ndarray):
        self._set_array_value(value)

    def _set_array_value(self, value: np.ndarray):
        if value is None:
            return

        if self._is_character_array():
            if self.base.kind == 1 and not np.issubdtype(value.dtype, np.bytes_):
                raise TypeError("Character strings must be bytes (S dtype)")
            if self.base.kind == 4 and not np.issubdtype(value.dtype, np.str_):
                raise TypeError("Unicode strings must be unicode (U dtype)")

        if value.ndim != self.ndims:
            raise ValueError(
                f"Wrong number of dimensions, got {value.ndim} expected {self.ndims}"
            )

        self._shape = tuple(value.shape)
        flat = (
            np.asfortranarray(value)
            .astype(self._array_dtype(value), copy=False)
            .ravel("F")
        )
        n = flat.size
        if self._is_character_array():
            strlen = int(self._array_dtype(value).itemsize)
            arr_type = (ctypes.c_char * strlen) * n
        else:
            arr_type = self.base.ctype * n
        self._ctype = arr_type()
        elem_size = ctypes.sizeof(self.base.ctype)
        if self._is_character_array():
            elem_size = int(self._array_dtype(value).itemsize)
        copy_array(
            flat.ctypes.data,
            ctypes.addressof(self._ctype),
            elem_size,
            n,
        )

    @property
    def ndims(self) -> int:
        return self._sym.properties.array_spec.rank


class ftype_assumed_shape(f_type, metaclass=ABCMeta):
    dtype = None  # type: ignore[assignment]
    ftype = None  # type: ignore[assignment]
    kind = None  # type: ignore[assignment]
    alloc_strategy: AllocStrategy = AllocStrategy.FORTRAN

    def __init__(self, value=None):
        self._value = value
        super().__init__()

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def base(self):
        return self._base()

    @property
    def ctype(self):
        if is_64bit():
            _index_t = ctypes.c_int64
            _size_t = ctypes.c_int64
        else:
            _index_t = ctypes.c_int32
            _size_t = ctypes.c_int32

        class _bounds14(ctypes.Structure):
            _fields_ = [
                ("stride", _index_t),
                ("lbound", _index_t),
                ("ubound", _index_t),
            ]

        class _dtype_type(ctypes.Structure):
            _fields_ = [
                ("elem_len", _size_t),
                ("version", ctypes.c_int32),
                ("rank", ctypes.c_byte),
                ("type", ctypes.c_byte),
                ("attribute", ctypes.c_ushort),
            ]

        class _fAllocArray(ctypes.Structure):
            _fields_ = [
                ("base_addr", ctypes.c_void_p),
                ("offset", _size_t),
                ("dtype", _dtype_type),
                ("span", _index_t),
                ("dims", _bounds14 * self.ndims),
            ]

        return _fAllocArray

    def __repr__(self):
        s = ",".join([":" for i in range(self.ndims)])
        return f"{self.ftype}(kind={self.kind})({s})"

    def _is_character_array(self) -> bool:
        return isinstance(self.base, ftype_character)

    def _character_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        strlen = int(self._sym.properties.typespec.charlen.value)
        if strlen <= 0 and value is not None:
            itemsize = int(np.asarray(value).dtype.itemsize)
            if self.base.kind == 4:
                strlen = max(itemsize // 4, 1)
            else:
                strlen = itemsize
        if strlen <= 0 and self._ctype.base_addr is not None:
            itemsize = int(self._ctype.dtype.elem_len)
            strlen = max(itemsize // 4, 1) if self.base.kind == 4 else itemsize
        if strlen <= 0:
            strlen = 1
        return np.dtype(f"U{strlen}") if self.base.kind == 4 else np.dtype(f"S{strlen}")

    def _array_dtype(self, value: np.ndarray | None = None) -> np.dtype:
        if self._is_character_array():
            return self._character_dtype(value)
        return self.base.dtype

    def _decode_kind4_flat(self, raw: bytes, elem_size: int, count: int) -> np.ndarray:
        values = []
        for i in range(count):
            chunk = raw[i * elem_size : (i + 1) * elem_size]
            utf8_bytes = bytes(chunk[j] for j in range(0, elem_size, 4))
            values.append(utf8_bytes.decode("utf-8").rstrip())
        return np.array(values, dtype=np.str_)

    def _kind4_utf8_len(self, value: np.ndarray) -> int:
        flat = np.asfortranarray(value).astype(np.str_, copy=False).ravel("F")
        if flat.size == 0:
            return 1
        return max(len(str(item).encode("utf-8")) for item in flat)

    def _write_kind4_array(self, value: np.ndarray, shape: tuple[int, ...]) -> None:
        flat = np.asfortranarray(value).astype(np.str_, copy=False).ravel("F")
        elem_size = int(self._ctype.dtype.elem_len)
        if elem_size <= 0:
            declared_len = int(self._sym.properties.typespec.charlen.value)
            if declared_len > 0:
                elem_size = declared_len * 4
            else:
                elem_size = self._kind4_utf8_len(value) * 4

        utf8_len = max(elem_size // 4, 1)
        base_addr = int(self._ctype.base_addr)
        for idx, item in enumerate(flat):
            encoded = str(item).encode("utf-8")
            if len(encoded) > utf8_len:
                encoded = encoded[:utf8_len]
            else:
                encoded = encoded + b" " * (utf8_len - len(encoded))

            raw = b"".join(bytes([byte, 0, 0, 0]) for byte in encoded)
            ctypes.memmove(base_addr + idx * elem_size, raw, len(raw))

        self._value = flat.reshape(shape, order="F")

    @property
    def value(self) -> Optional[np.ndarray]:
        if self._ctype.base_addr is None:
            return None

        shape_list = []
        for i in range(self.ndims):
            shape_list.append(
                self._ctype.dims[i].ubound - self._ctype.dims[i].lbound + 1
            )

        shape = tuple(shape_list)
        count = int(np.prod(shape))

        if self._is_character_array() and self.base.kind == 4:
            elem_size = int(self._ctype.dtype.elem_len)
            raw = ctypes.string_at(self._ctype.base_addr, count * elem_size)
            flat = self._decode_kind4_flat(raw, elem_size, count)
            self._value = flat.reshape(shape, order="F")
            return self._value

        array = np.zeros(shape, dtype=self._array_dtype(), order="F")
        elem_size = ctypes.sizeof(self.base.ctype)
        if self._is_character_array():
            elem_size = int(self._array_dtype().itemsize)

        copy_array(
            self._ctype.base_addr,
            array.ctypes.data,
            elem_size,
            count,
        )
        self._value = array
        return array

    @value.setter
    def value(self, value: np.ndarray):
        self._set_descriptor_value(value)

    def _set_descriptor_value(self, value: np.ndarray):
        if value is None:
            return

        if self._is_character_array():
            if self.base.kind == 1 and not np.issubdtype(value.dtype, np.bytes_):
                raise TypeError("Character strings must be bytes (S dtype)")
            if self.base.kind == 4 and not np.issubdtype(value.dtype, np.str_):
                raise TypeError("Unicode strings must be unicode (U dtype)")

        shape = np.shape(value)

        if (
            self._is_character_array()
            and self._sym.properties.typespec.charlen.value <= 0
        ):
            if self.base.kind == 4:
                strlen = self._kind4_utf8_len(value)
            else:
                strlen = int(np.asarray(value).dtype.itemsize)
            if strlen <= 0:
                strlen = 1
            self.base.value = b" " * strlen

        self._allocate(shape)

        if self._is_character_array() and self.base.kind == 4:
            self._write_kind4_array(value, shape)
            return

        self._value = (
            np.asfortranarray(value)
            .astype(self._array_dtype(value), copy=False)
            .ravel("F")
        )
        elem_size = ctypes.sizeof(self.base.ctype)
        if self._is_character_array():
            elem_size = int(self._array_dtype(value).itemsize)
        copy_array(
            self._value.ctypes.data,
            self._ctype.base_addr,
            elem_size,
            int(np.prod(shape)),
        )

    def _allocate(self, shape):
        code = self.base.allocate(shape)
        args = CompileArgs()
        if self.base.extra_fflags:
            args.FFLAGS = f"{args.FFLAGS} {self.base.extra_fflags}".strip()

        cache_key = (code.strhash(), str(args))
        if cache_key in _ALLOCATOR_CACHE:
            lib, sub = _ALLOCATOR_CACHE[cache_key]
        else:
            comp = Compile(code.as_module(), name=code.strhash())
            if not comp.compile(args=args):
                raise AllocationError("Failed to allocate array")

            lib = comp.platform.load_library(comp.library_filename)
            name = f"__{comp.name}_MOD_alloc"
            sub = getattr(lib, name)
            _ALLOCATOR_CACHE[cache_key] = (lib, sub)

        sub(ctypes.byref(self._ctype))

        # Did allocation work?
        if self._ctype.base_addr is None:
            raise ValueError("Allocation failed")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._sym.properties.array_spec.pyshape

    @property
    def ndims(self) -> int:
        return self._sym.properties.array_spec.rank


class ftype_assumed_rank(ftype_assumed_shape, metaclass=ABCMeta):
    """Fortran assumed-rank dummy argument: dimension(..).

    The rank is not known from the symbol metadata and must be derived from
    the numpy value passed at runtime.
    """

    def __init__(self, value=None):
        self._ndims = 0
        super().__init__(value=None)
        if value is not None:
            self.value = value

    @property
    def value(self) -> Optional[np.ndarray]:
        return super().value

    @value.setter
    def value(self, value: np.ndarray):
        if value is None:
            return

        shape = np.shape(value)
        self._ndims = len(shape)
        self._ctype = self.ctype()
        self._set_descriptor_value(value)

    @property
    def ndims(self) -> int:
        return self._ndims
