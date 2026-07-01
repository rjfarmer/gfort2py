# SPDX-License-Identifier: GPL-2.0+

import ctypes
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple, Type

import gfModParser as gf
import numpy as np

try:
    import pyquadp as pyq  # type: ignore[import-not-found]

    PYQ_IMPORTED = True
except ImportError:
    PYQ_IMPORTED = False

from ..compilation import Compile, CompileArgs, Modulise
from ..utils import copy_array, is_64bit
from .base import AllocStrategy, f_type
from .character import ftype_character
from .numpy_convert import to_numpy_array_with_dtype


class AllocationError(Exception):
    pass


# Cache compiled allocator entry points by (generated module name, compile args)
# so repeated array marshaling avoids spawning the compiler each call.
_ALLOCATOR_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


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

    def _quad_raw_array(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = (
            np.asfortranarray(value)
            .ravel(order="F")
            .astype(self.base.dtype, copy=False)
        )
        return flat

    @property
    def value(self) -> np.ndarray:
        if self._is_character_array():
            if self.base.kind == 4:
                elem_size = ctypes.sizeof(self._ctype._type_)
                raw = ctypes.string_at(
                    ctypes.addressof(self._ctype), self.size * elem_size
                )
                decoded_values = []
                for i in range(self.size):
                    chunk = raw[i * elem_size : (i + 1) * elem_size]
                    utf8_bytes = bytes(chunk[j] for j in range(0, elem_size, 4))
                    decoded_values.append(utf8_bytes.decode("utf-8").rstrip())
                self._value = np.array(decoded_values, dtype=np.str_).reshape(
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

        if self.base.kind == 16:
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )

            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), self.size * elem_size)
            quad_values = [
                self.base.pytype.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(self.size)
            ]
            self._value = np.array(quad_values, dtype=object).reshape(
                self.shape, order="F"
            )
            return self._value

        self._value = to_numpy_array_with_dtype(
            np.ctypeslib.as_array(self._ctype).reshape(self.shape, order="F"),
            self._array_dtype(),
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

                    raw_utf8 = b"".join(bytes([byte, 0, 0, 0]) for byte in encoded)
                    ctypes.memmove(base_addr + idx * elem_size, raw_utf8, len(raw_utf8))
                return

            elem_size = int(self._array_dtype(value).itemsize)

        if self.base.kind == 16:
            raw_array = self._quad_raw_array(value)
            copy_array(
                raw_array.ctypes.data,
                ctypes.addressof(self._ctype),
                elem_size,
                self.size,
            )
            return

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

    def _is_quad_int_array(self) -> bool:
        return self.base.ftype == "integer" and self.base.kind == 16

    def _is_quad_real_array(self) -> bool:
        return self.base.ftype == "real" and self.base.kind == 16

    def _is_quad_complex_array(self) -> bool:
        return self.base.ftype == "complex" and self.base.kind == 16

    def _quad_real_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

    def _quad_int_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

    def _quad_complex_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

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

        if self._is_quad_real_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            n = len(self._ctype)
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), n * elem_size)
            quad_real_values = [
                pyq.qfloat.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(n)
            ]
            arr = np.array(quad_real_values, dtype=object)
            if self._shape is not None:
                arr = arr.reshape(self._shape, order="F")
            return arr

        if self._is_quad_int_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            n = len(self._ctype)
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), n * elem_size)
            quad_int_values = [
                pyq.qint.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(n)
            ]
            arr = np.array(quad_int_values, dtype=object)
            if self._shape is not None:
                arr = arr.reshape(self._shape, order="F")
            return arr

        if self._is_quad_complex_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            n = len(self._ctype)
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(ctypes.addressof(self._ctype), n * elem_size)
            quad_complex_values = [
                pyq.qcmplx.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(n)
            ]
            arr = np.array(quad_complex_values, dtype=object)
            if self._shape is not None:
                arr = arr.reshape(self._shape, order="F")
            return arr

        arr = to_numpy_array_with_dtype(
            np.ctypeslib.as_array(self._ctype),
            self._array_dtype(),
        )
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
        if (
            self._is_quad_real_array()
            or self._is_quad_complex_array()
            or self._is_quad_int_array()
        ):
            flat = np.asfortranarray(value).ravel("F")
        else:
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

        if self._is_quad_int_array():
            raw = self._quad_int_raw_flat(value)
            copy_array(
                raw.ctypes.data,
                ctypes.addressof(self._ctype),
                elem_size,
                n,
            )
            return

        if self._is_quad_real_array():
            raw = self._quad_real_raw_flat(value)
            copy_array(
                raw.ctypes.data,
                ctypes.addressof(self._ctype),
                elem_size,
                n,
            )
            return

        if self._is_quad_complex_array():
            raw = self._quad_complex_raw_flat(value)
            copy_array(
                raw.ctypes.data,
                ctypes.addressof(self._ctype),
                elem_size,
                n,
            )
            return

        copy_array(
            flat.ctypes.data,
            ctypes.addressof(self._ctype),
            elem_size,
            n,
        )

    @property
    def ndims(self) -> int:
        return self._sym.properties.array_spec.rank


# Cache of the GFortran array-descriptor ctypes.Structure subclass, keyed by
# (ndims, is_64bit()). The descriptor layout depends only on these two things,
# so it can be built once and reused. Rebuilding a fresh Structure subclass on
# every call leaks memory: CPython's ctypes never frees Structure subclasses.
_ASSUMED_SHAPE_CTYPE_CACHE: dict = {}


class ftype_assumed_shape(f_type, metaclass=ABCMeta):
    dtype = None  # type: ignore[assignment]
    ftype = None  # type: ignore[assignment]
    kind = None  # type: ignore[assignment]
    alloc_strategy: AllocStrategy = AllocStrategy.FORTRAN

    def __init__(self, value=None):
        self._value = value
        self._alloc_cache_key: tuple[str, str] | None = None
        self._base_obj = self._base()
        super().__init__()

    def _deallocate_subroutine_text(self, ndims: int) -> str:
        dims = ",".join([":"] * ndims)

        if self._is_character_array():
            decl = (
                f"{self.base.ftype}(kind={self.base.kind},len={self.base.strlen}),"
                f"allocatable,dimension({dims}), intent(inout) :: x"
            )
        else:
            decl = (
                f"{self.base.ftype}(kind={self.base.kind}),"
                f"allocatable,dimension({dims}), intent(inout) :: x"
            )

        return f"""
        subroutine dealloc(x)
        {decl}
        if(allocated(x)) deallocate(x)
        end subroutine dealloc
        """

    @abstractmethod
    def _base(self):
        raise NotImplementedError

    @property
    def base(self):
        return self._base_obj

    @property
    def ctype(self):
        _key = (self.ndims, is_64bit())
        _cached = _ASSUMED_SHAPE_CTYPE_CACHE.get(_key)
        if _cached is not None:
            return _cached

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

        _ASSUMED_SHAPE_CTYPE_CACHE[_key] = _fAllocArray
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

    def _is_quad_int_array(self) -> bool:
        return self.base.ftype == "integer" and self.base.kind == 16

    def _is_quad_real_array(self) -> bool:
        return self.base.ftype == "real" and self.base.kind == 16

    def _is_quad_complex_array(self) -> bool:
        return self.base.ftype == "complex" and self.base.kind == 16

    def _quad_real_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

    def _quad_int_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

    def _quad_complex_raw_flat(self, value: np.ndarray) -> np.ndarray:
        if not PYQ_IMPORTED:
            raise ValueError("Please install pyQuadp to handle quad precision numbers")

        flat = np.asfortranarray(value).ravel("F").astype(self.base.dtype, copy=False)
        return flat

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

        if self._is_quad_int_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(self._ctype.base_addr, count * elem_size)
            quad_int_values = [
                pyq.qint.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(count)
            ]
            self._value = np.array(quad_int_values, dtype=object).reshape(
                shape, order="F"
            )
            return self._value

        if self._is_quad_real_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(self._ctype.base_addr, count * elem_size)
            quad_real_values = [
                pyq.qfloat.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(count)
            ]
            self._value = np.array(quad_real_values, dtype=object).reshape(
                shape, order="F"
            )
            return self._value

        if self._is_quad_complex_array():
            if not PYQ_IMPORTED:
                raise ValueError(
                    "Please install pyQuadp to handle quad precision numbers"
                )
            elem_size = ctypes.sizeof(self.base.ctype)
            raw = ctypes.string_at(self._ctype.base_addr, count * elem_size)
            quad_complex_values = [
                pyq.qcmplx.from_bytes(raw[i * elem_size : (i + 1) * elem_size])
                for i in range(count)
            ]
            self._value = np.array(quad_complex_values, dtype=object).reshape(
                shape, order="F"
            )
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

        needs_alloc = self._ctype.base_addr is None
        if not needs_alloc:
            current_shape = tuple(
                self._ctype.dims[i].ubound - self._ctype.dims[i].lbound + 1
                for i in range(self.ndims)
            )
            needs_alloc = current_shape != tuple(shape)

        if (
            self._is_character_array()
            and not needs_alloc
            and self._sym.properties.typespec.charlen.value <= 0
        ):
            desired_elem_len = int(self._array_dtype(value).itemsize)
            current_elem_len = int(self._ctype.dtype.elem_len)
            needs_alloc = desired_elem_len != current_elem_len

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

        if needs_alloc:
            self._allocate(shape)

        if self._is_character_array() and self.base.kind == 4:
            self._write_kind4_array(value, shape)
            return

        if self._is_quad_real_array():
            self._value = np.asfortranarray(value).ravel("F")
            raw = self._quad_real_raw_flat(value)
            elem_size = ctypes.sizeof(self.base.ctype)
            copy_array(
                raw.ctypes.data,
                self._ctype.base_addr,
                elem_size,
                int(np.prod(shape)),
            )
            return

        if self._is_quad_complex_array():
            self._value = np.asfortranarray(value).ravel("F")
            raw = self._quad_complex_raw_flat(value)
            elem_size = ctypes.sizeof(self.base.ctype)
            copy_array(
                raw.ctypes.data,
                self._ctype.base_addr,
                elem_size,
                int(np.prod(shape)),
            )
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
        alloc_code = self.base.allocate(shape)
        code = Modulise(
            "\n".join([alloc_code.text, self._deallocate_subroutine_text(len(shape))])
        )
        args = CompileArgs()
        if self.base.extra_fflags:
            args.FFLAGS = f"{args.FFLAGS} {self.base.extra_fflags}".strip()

        cache_key = (code.strhash(), str(args))
        self._alloc_cache_key = cache_key
        if cache_key in _ALLOCATOR_CACHE:
            lib, sub, _dealloc = _ALLOCATOR_CACHE[cache_key]
        else:
            comp = Compile(code.as_module(), name=code.strhash())
            if not comp.compile(args=args):
                raise AllocationError("Failed to allocate array")

            lib = comp.platform.load_library(comp.library_filename)
            alloc_name = f"__{comp.name}_MOD_alloc"
            dealloc_name = f"__{comp.name}_MOD_dealloc"
            sub = getattr(lib, alloc_name)
            dealloc_sub = getattr(lib, dealloc_name)
            _ALLOCATOR_CACHE[cache_key] = (lib, sub, dealloc_sub)

        sub(ctypes.byref(self._ctype))

        # Did allocation work?
        if self._ctype.base_addr is None:
            raise ValueError("Allocation failed")

    def release(self) -> None:
        if self._ctype.base_addr is None:
            return

        if (
            self._alloc_cache_key is not None
            and self._alloc_cache_key in _ALLOCATOR_CACHE
        ):
            _lib, _alloc_sub, dealloc_sub = _ALLOCATOR_CACHE[self._alloc_cache_key]
            dealloc_sub(ctypes.byref(self._ctype))

        self._ctype.base_addr = None

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
